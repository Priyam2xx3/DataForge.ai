from sqlalchemy import Column, Integer, String, DateTime, Float
from database import Base
import datetime

class DatasetMetadata(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    filepath = Column(String, nullable=False)
    total_rows = Column(Integer, default=0)
    total_cols = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
