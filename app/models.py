from sqlalchemy import Column, Integer, String, Float, LargeBinary, DateTime
from datetime import datetime
from app.database import Base

class TrainingLog(Base):
    __tablename__ = "training_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    project_name = Column(String)
    model_name = Column(String)
    train_accuracy = Column(Float)
    val_accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    model_blob = Column(LargeBinary)
    class_names = Column(String)  # Comma-separated class labels
    timestamp = Column(DateTime, default=datetime.utcnow)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    project_name = Column(String)
    predicted_class = Column(String)
    confidence_score = Column(Float)
    image_path = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
