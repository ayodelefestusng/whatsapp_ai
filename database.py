from sqlalchemy import create_engine, Column, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./whatsapp_bot.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserState(Base):
    __tablename__ = "user_states"
    
    # Use the WhatsApp phone number as the unique ID
    phone_number = Column(String, primary_key=True, index=True)
    state = Column(String, default="idle")  # e.g., 'leave_application'
    step = Column(String, nullable=True)     # e.g., 'awaiting_date'
    data = Column(JSON, default={})          # Store collected info here

# Create the table
Base.metadata.create_all(bind=engine)