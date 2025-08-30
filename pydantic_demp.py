from pydantic import BaseModel
from typing import Optional
class Student(BaseModel):
    name:str='chinu'
    age:Optional[int]= None
new_student = {'age':'20'}
student = Student(**new_student)
print(student)