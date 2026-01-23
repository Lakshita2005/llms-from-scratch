from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Anya'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')  # setting constraints "lt= less than equals to" and so on..., u can add RegEx as well 

     

# new_student = {'age': '22'} # Coercing: despite of this mis-typed value pydantic is smart engh to convert it into its original type, so here '32' will be typecasted to int
new_student = {'age': 32,'email':'abc@gmail.com'}
student = Student(**new_student)
student_dict = dict(student)
student_json = student.model_dump_json()

print(student)
print(student_dict)
print(student_json)