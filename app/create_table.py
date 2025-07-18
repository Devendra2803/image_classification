from app.database import Base, engine
from app import models

# Create all tables defined in models.py
def create_tables():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    create_tables()
    print("✅ Database tables created successfully.")
