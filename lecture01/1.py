class StudentManager:
    def __init__(self):
        self.students = []
    def add_student(self,student):
        self.students.append(student)
    def display_all_students(self):
        for i in self.students:
            print(i)

class Student:
    i = 0
    student_id = []
    name = []
    age = []
    def display_info(self):
        print("ID:" + self.student_id[self.i] + " / 이름: " + self.name[self.i] + " / 나이: " + self.age[self.i] + "살")
        self.i += 1

a = Student()
b = StudentManager()
j = 1
y = 1
print("현재 등록된 학생 목록:")
while j <= 3:
    T = input("띄어쓰기로 구분해서 학생의 번호 이름 나이를 입력해 주세요.\n")
    T = T.split(' ')
    a.student_id.append(T[0])
    a.name.append(T[1])
    a.age.append(T[2])
    b.students.append("ID:" + a.student_id[a.i] + " / 이름: " + a.name[a.i] + " / 나이: " + a.age[a.i] + "살")
    T = []
    j += 1
while y <= 3:
    a.display_info()
    y += 1
T = input("추가하실 학생의 번호 이름 나이 순서로 띄어쓰기로 구분해서 입력해 주세요.\n")
T = T.split(' ')
a.student_id.append(T[0])
a.name.append(T[1])
a.age.append(T[2])
print("학번 4번 학생 추가 후:")
b.students.append("ID:" + a.student_id[a.i] + " / 이름: " + a.name[a.i] + " / 나이: " + a.age[a.i] + "살")
b.display_all_students()