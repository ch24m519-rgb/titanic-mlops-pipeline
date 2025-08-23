from pydantic import BaseModel
from typing import Optional

class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: Optional[float] = None
    SibSp: int
    Parch: int
    Fare: Optional[float] = None
    Embarked: Optional[str] = None

