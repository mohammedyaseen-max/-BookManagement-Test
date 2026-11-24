import os
import psycopg2

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://bookuser:bookpassword@127.0.0.1:5432/bookdb"
)

# قائمة الكتب المراد إضافتها (اسم الكتاب، اسم الكاتب)
sample_books = [
    ("The Great Gatsby", "F. Scott Fitzgerald"),
    ("To Kill a Mockingbird", "Harper Lee"),
    ("1984", "George Orwell"),
    ("Pride and Prejudice", "Jane Austen"),
    ("The Catcher in the Rye", "J.D. Salinger"),
    ("The Hobbit", "J.R.R. Tolkien"),
    ("Fahrenheit 451", "Ray Bradbury"),
    ("Jane Eyre", "Charlotte Brontë"),
    ("Animal Farm", "George Orwell"),
    ("Brave New World", "Aldous Huxley"),
    ("The Chronicles of Narnia", "C.S. Lewis"),
    ("The Lord of the Rings", "J.R.R. Tolkien"),
    ("Crime and Punishment", "Fyodor Dostoevsky"),
    ("Wuthering Heights", "Emily Brontë"),
    ("The Alchemist", "Paulo Coelho")
]

def seed_database():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # إدخال الكتب إلى الجدول
        for title, author in sample_books:
            cursor.execute(
                "INSERT INTO books (title, author) VALUES (%s, %s)",
                (title, author)
            )
            
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Done! Successfully inserted {len(sample_books)} books into PostgreSQL.")
    except Exception as e:
        print(f"Error while inserting books: {e}")

if __name__ == "__main__":
    seed_database()