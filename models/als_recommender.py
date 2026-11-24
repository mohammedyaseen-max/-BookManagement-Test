#!/usr/bin/env python3
"""Train/evaluate a Spark MLlib ALS book recommender on cleaned Book-Crossing data.

Expected input: data/processed/ratings_enriched (Parquet), created by clean_with_spark.py.
Outputs are written beneath analysis/results/als_recommender.
"""
from pathlib import Path
import json
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

ROOT = Path(__file__).resolve().parents[1]
INPUT = str(ROOT / 'data' / 'processed' / 'ratings_enriched')
OUTPUT = str(ROOT / 'analysis' / 'results' / 'als_recommender')

spark = (SparkSession.builder
         .appName('BookCrossing-ALS-Recommender')
         .config('spark.sql.shuffle.partitions', '8')
         .getOrCreate())
spark.sparkContext.setLogLevel('WARN')

try:
    df = spark.read.parquet(INPUT)
    required = {'User-ID', 'ISBN', 'Book-Rating'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {INPUT}: {sorted(missing)}; found {df.columns}')

    # Explicit ratings only: Book-Crossing rating 0 means implicit/unspecified,
    # so it is excluded from this explicit-rating prediction task.
    ratings = (df.select(
                    F.trim(F.col('User-ID').cast('string')).alias('user_id_text'),
                    F.trim(F.col('ISBN').cast('string')).alias('isbn'),
                    F.col('Book-Rating').cast('float').alias('rating'))
               .filter(F.col('user_id_text').rlike('^[0-9]+$'))
               .filter(F.col('isbn').isNotNull() & (F.length('isbn') > 0))
               .filter(F.col('rating').between(1, 10))
               .withColumn('user_id', F.col('user_id_text').cast('int'))
               .filter(F.col('user_id').isNotNull())
               .select('user_id', 'isbn', 'rating')
               .dropDuplicates(['user_id', 'isbn']))

    count = ratings.count()
    if count < 100:
        raise ValueError(f'Only {count} usable explicit ratings found; check processed data.')

    # Create stable numeric item IDs for ALS. Fit the indexer on the interaction
    # set before the split so test items have known encodings; ALS itself only
    # learns from the training split. Unseen-user/item predictions are dropped.
    indexer = StringIndexer(inputCol='isbn', outputCol='item_id', handleInvalid='skip')
    indexed = (indexer.fit(ratings).transform(ratings)
               .select('user_id', F.col('item_id').cast('int').alias('item_id'), 'rating'))

    train, test = indexed.randomSplit([0.8, 0.2], seed=42)
    train = train.cache()
    test = test.cache()
    train_count, test_count = train.count(), test.count()
    if train_count == 0 or test_count == 0:
        raise ValueError(f'Empty train/test split: train={train_count}, test={test_count}')

    als = ALS(userCol='user_id', itemCol='item_id', ratingCol='rating',
              rank=10, maxIter=10, regParam=0.1,
              coldStartStrategy='drop', nonnegative=True, seed=42)
    model = als.fit(train)
    predictions = model.transform(test).cache()
    evaluable_count = predictions.count()

    evaluator_rmse = RegressionEvaluator(labelCol='rating', predictionCol='prediction', metricName='rmse')
    evaluator_mae = RegressionEvaluator(labelCol='rating', predictionCol='prediction', metricName='mae')
    als_rmse = float(evaluator_rmse.evaluate(predictions)) if evaluable_count else None
    als_mae = float(evaluator_mae.evaluate(predictions)) if evaluable_count else None

    # Simple benchmark: predict the training-set global mean for every test row.
    global_mean = float(train.agg(F.avg('rating').alias('m')).first()['m'])
    baseline = test.withColumn('prediction', F.lit(global_mean))
    baseline_rmse = float(evaluator_rmse.evaluate(baseline))
    baseline_mae = float(evaluator_mae.evaluate(baseline))

    metrics = {
        'task': 'Explicit book-rating prediction',
        'input_parquet': 'data/processed/ratings_enriched',
        'rating_filter': 'Book-Rating 1 through 10; rating 0 excluded',
        'usable_interactions': int(count),
        'train_rows': int(train_count),
        'test_rows': int(test_count),
        'als_evaluable_test_rows_after_cold_start_drop': int(evaluable_count),
        'train_global_mean_rating': global_mean,
        'als': {'rank': 10, 'maxIter': 10, 'regParam': 0.1, 'rmse': als_rmse, 'mae': als_mae},
        'global_mean_baseline': {'rmse': baseline_rmse, 'mae': baseline_mae},
        'note': 'Lower RMSE/MAE is better. ALS metrics use only test rows for which ALS produced a prediction; the baseline uses all test rows. Compare coverage as well as error.'
    }

    out = Path(OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    metric_rows = [
        ('ALS', als_rmse, als_mae, evaluable_count),
        ('Global-mean baseline', baseline_rmse, baseline_mae, test_count),
    ]
    spark.createDataFrame(metric_rows, ['model', 'rmse', 'mae', 'evaluated_test_rows']) \
         .coalesce(1).write.mode('overwrite').option('header', True).csv(OUTPUT + '/metrics_csv')

    # Save a compact set of top recommendations for the most active training user.
    active = train.groupBy('user_id').count().orderBy(F.desc('count'), F.asc('user_id')).first()
    if active:
        chosen_user = int(active['user_id'])
        recs = model.recommendForUserSubset(
            spark.createDataFrame([(chosen_user,)], ['user_id']), 10)
        item_lookup = indexer.fit(ratings).labels
        from pyspark.sql.types import StructType, StructField, IntegerType, StringType
        lookup = spark.createDataFrame(
            [(i, isbn) for i, isbn in enumerate(item_lookup)],
            schema=StructType([StructField('item_id', IntegerType(), False),
                               StructField('isbn', StringType(), False)])
        )
        exploded = (recs.select('user_id', F.explode('recommendations').alias('rec'))
                    .select('user_id', F.col('rec.item_id').alias('item_id'),
                            F.col('rec.rating').alias('predicted_rating'))
                    .join(lookup, 'item_id', 'left')
                    .orderBy(F.desc('predicted_rating')))
        exploded.coalesce(1).write.mode('overwrite').option('header', True).csv(OUTPUT + '/sample_top10_recommendations')
        metrics['sample_recommendation_user_id'] = chosen_user
        with open(out / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    print('\n=== ALS MODEL EVALUATION ===')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f'\nOutputs saved to: {OUTPUT}')
    predictions.unpersist()
    train.unpersist()
    test.unpersist()
finally:
    spark.stop()
