以下是一个简要而全面的C++语言教程，涵盖核心概念和关键特性，适合快速复习或入门。内容尽量精炼，覆盖基础到中级知识点，帮助你快速掌握C++。

---
# C++语言教程（扩展版）

## 目录
- [1. 简介](#1-简介)
- [2. 基础语法](#2-基础语法)
  - [2.1 程序结构](#21-程序结构)
  - [2.2 变量与数据类型](#22-变量与数据类型)
  - [2.3 输入输出](#23-输入输出)
  - [2.4 运算符](#24-运算符)
  - [2.5 控制结构](#25-控制结构)
- [3. 函数](#3-函数)
- [4. 数组与字符串](#4-数组与字符串)
- [5. 指针与引用](#5-指针与引用)
- [6. 面向对象编程（OOP）](#6-面向对象编程oop)
  - [6.1 类与对象](#61-类与对象)
  - [6.2 继承](#62-继承)
  - [6.3 多态](#63-多态)
  - [6.4 运算符重载](#64-运算符重载)
  - [6.5 友元与静态成员](#65-友元与静态成员)
- [7. 标准模板库（STL）](#7-标准模板库stl)
  - [7.1 容器](#71-容器)
  - [7.2 迭代器](#72-迭代器)
  - [7.3 算法](#73-算法)
  - [7.4 自定义比较器](#74-自定义比较器)
  - [7.5 性能与选择](#75-性能与选择)
- [8. 模板（泛型编程）](#8-模板泛型编程)
- [9. 异常处理](#9-异常处理)
- [10. 其他高级特性](#10-其他高级特性)
- [11. 最佳实践](#11-最佳实践)
- [12. 学习资源](#12-学习资源)
  - [12.1 书籍](#121-书籍)
  - [12.2 在线平台](#122-在线平台)
  - [12.3 工具](#123-工具)
  - [12.4 社区与论坛](#124-社区与论坛)
  - [12.5 实践项目](#125-实践项目)
  - [12.6 标准与更新](#126-标准与更新)

---

## 1. 简介
C++ 是一种高效、通用、面向对象的编程语言，由Bjarne Stroustrup在1980年代开发，扩展自C语言。它支持过程式、面向对象和泛型编程，广泛应用于系统开发、游戏开发、嵌入式系统和高性能计算。

---

## 2. 基础语法

### 2.1 程序结构
```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
```
- **#include**：引入头文件，如`<iostream>`用于输入输出。
- **using namespace std;**：简化标准库命名空间的使用（生产环境慎用）。
- **main()**：程序入口，返回`int`。

### 2.2 变量与数据类型
- 基本类型：`int`, `double`, `float`, `char`, `bool`。
- 复合类型：数组、指针、引用、结构体、枚举。
```cpp
int x = 10;
double pi = 3.14159;
char c = 'A';
bool flag = true;
string name = "Alice"; // 需包含<string>
```

### 2.3 输入输出
```cpp
#include <iostream>
int age;
cout << "Enter your age: ";
cin >> age;
cout << "You are " << age << " years old." << endl;
```

### 2.4 运算符
- 算术：`+`, `-`, `*`, `/`, `%`。
- 关系：`==`, `!=`, `<`, `>`, `<=`, `>=`。
- 逻辑：`&&`, `||`, `!`。
- 位运算：`&`, `|`, `^`, `~`, `<<`, `>>`。
- 赋值：`=`, `+=`, `-=`, `*=`, `/=`, `%=`。

### 2.5 控制结构
- **if-else**：
```cpp
if (x > 0) {
    cout << "Positive";
} else if (x == 0) {
    cout << "Zero";
} else {
    cout << "Negative";
}
```

- **循环**：
```cpp
for (int i = 0; i < 5; i++) {
    cout << i << " ";
}
int i = 0;
while (i < 5) {
    cout << i << " ";
    i++;
}
do {
    cout << i << " ";
    i++;
} while (i < 5);
```

- **switch**：
```cpp
switch (x) {
    case 1: cout << "One"; break;
    case 2: cout << "Two"; break;
    default: cout << "Other"; break;
}
```

---

## 3. 函数
```cpp
int add(int a, int b) {
    return a + b;
}
int main() {
    cout << add(3, 4); // 输出7
    return 0;
}
```
- **函数重载**：
```cpp
int add(int a, int b) { return a + b; }
double add(double a, double b) { return a + b; }
```
- **默认参数**：
```cpp
void print(int x, int y = 10) { cout << x << ", " << y; }
```
- **内联函数**：
```cpp
inline int square(int x) { return x * x; }
```

---

## 4. 数组与字符串
- **数组**：
```cpp
int arr[5] = {1, 2, 3, 4, 5};
cout << arr[0]; // 输出1
```
- **字符串**（`<string>`）：
```cpp
string str = "Hello";
str += ", World!";
cout << str.length(); // 输出12
```
- **C风格字符串**：
```cpp
char cstr[] = "Hello";
```

---

## 5. 指针与引用
- **指针**：
```cpp
int x = 10;
int* ptr = &x;
cout << *ptr; // 输出10
```
- **引用**：
```cpp
int a = 5;
int& ref = a;
ref = 10;
cout << a; // 输出10
```
- **动态内存**：
```cpp
int* ptr = new int(5);
delete ptr;
```

---

## 6. 面向对象编程（OOP）
C++的OOP特性包括封装、继承、多态和抽象，广泛用于模块化设计。

### 6.1 类与对象
类是用户定义的数据类型，封装数据和行为。
```cpp
class Person {
public:
    string name;
    int age;
    Person(string n, int a) : name(n), age(a) {} // 构造函数
    ~Person() { cout << "Person destroyed" << endl; } // 析构函数
    void introduce() const { // const成员函数
        cout << "I am " << name << ", " << age << " years old." << endl;
    }
private:
    double salary; // 私有成员
public:
    void setSalary(double s) { salary = s; } // 访问器
    double getSalary() const { return salary; }
};

int main() {
    Person p("Alice", 25);
    p.setSalary(50000);
    p.introduce(); // 输出: I am Alice, 25 years old.
    cout << "Salary: " << p.getSalary() << endl;
    return 0;
}
```
- **封装**：通过`private`和`public`控制访问，保护数据。
- **构造函数**：初始化对象，允许多个重载版本。
- **析构函数**：对象销毁时自动调用，释放资源。
- **const成员函数**：保证不修改对象状态。

### 6.2 继承
继承实现代码复用和层级关系。
```cpp
class Student : public Person {
public:
    int studentID;
    Student(string n, int a, int id) : Person(n, a), studentID(id) {}
    void study() { cout << name << " is studying." << endl; }
};

int main() {
    Student s("Bob", 20, 12345);
    s.introduce(); // 继承自Person
    s.study(); // 输出: Bob is studying.
    return 0;
}
```
- **访问修饰符**：`public`继承保持基类访问权限，`protected`和`private`继承限制访问。
- **构造函数调用**：派生类构造函数需显式调用基类构造函数。

### 6.3 多态
多态通过虚函数实现运行时动态行为。
```cpp
class Animal {
public:
    virtual void speak() const { cout << "Generic sound" << endl; } // 虚函数
    virtual ~Animal() = default; // 虚析构函数
};

class Dog : public Animal {
public:
    void speak() const override { cout << "Woof!" << endl; }
};

class Cat : public Animal {
public:
    void speak() const override { cout << "Meow!" << endl; }
};

int main() {
    Animal* animals[] = {new Dog(), new Cat()};
    for (Animal* a : animals) {
        a->speak(); // 输出: Woof! Meow!
        delete a;
    }
    return 0;
}
```
- **虚函数**：通过`virtual`关键字实现动态绑定。
- **纯虚函数与抽象类**：
```cpp
class Shape {
public:
    virtual double area() const = 0; // 纯虚函数
    virtual ~Shape() = default;
};
```
- **虚析构函数**：确保派生类对象通过基类指针删除时正确释放资源。

### 6.4 运算符重载
自定义运算符行为，如`+`或`<<`。
```cpp
class Vector2D {
public:
    double x, y;
    Vector2D(double x_, double y_) : x(x_), y(y_) {}
    Vector2D operator+(const Vector2D& other) const {
        return Vector2D(x + other.x, y + other.y);
    }
};

int main() {
    Vector2D v1(1, 2), v2(3, 4);
    Vector2D v3 = v1 + v2; // 输出: (4, 6)
    cout << "(" << v3.x << ", " << v3.y << ")" << endl;
    return 0;
}
```

### 6.5 友元与静态成员
- **友元**：允许外部函数或类访问私有成员。
```cpp
class Box {
    friend void openBox(Box& b); // 友元函数
private:
    int content;
};
```
- **静态成员**：类级成员，共享于所有对象。
```cpp
class Counter {
public:
    static int count; // 静态成员
    Counter() { count++; }
};
int Counter::count = 0; // 初始化
```

---

## 7. 标准模板库（STL）
STL是C++的核心库，提供容器、算法和迭代器，需包含相应头文件（如`<vector>`, `<algorithm>`）。

### 7.1 容器
STL提供多种容器，分为序列容器、关联容器和容器适配器。

- **序列容器**：
  - `vector`（动态数组，随机访问快，尾部插入高效）。
  ```cpp
  #include <vector>
  vector<int> vec = {1, 2, 3};
  vec.push_back(4); // 尾部添加
  vec[0] = 10; // 随机访问
  ```
  - `array`（固定大小数组，C++11起，`<array>`）。
  ```cpp
  #include <array>
  array<int, 3> arr = {1, 2, 3};
  ```
  - `list`（双向链表，插入删除快，`<list>`）。
  ```cpp
  #include <list>
  list<int> lst = {1, 2, 3};
  lst.push_front(0); // 头部添加
  ```
  - `deque`（双端队列，`<deque>`）。

- **关联容器**：
  - `set`（有序唯一集合，`<set>`）。
  ```cpp
  #include <set>
  set<int> s = {3, 1, 2};
  s.insert(4); // 自动排序
  ```
  - `map`（键值对，键唯一，`<map>`）。
  ```cpp
  #include <map>
  map<string, int> m;
  m["Alice"] = 25;
  cout << m["Alice"]; // 输出25
  ```
  - `multiset`, `multimap`（允许重复键）。

- **无序容器**（C++11起，哈希表实现）：
  - `unordered_set`, `unordered_map`（`<unordered_set>`, `<unordered_map>`）。

- **容器适配器**：
  - `stack`（栈，`<stack>`）。
  - `queue`（队列，`<queue>`）。
  - `priority_queue`（优先队列，`<queue>`）。
  ```cpp
  #include <stack>
  stack<int> s;
  s.push(1);
  s.pop();
  ```

### 7.2 迭代器
迭代器用于遍历容器，类似指针。
```cpp
vector<int> vec = {1, 2, 3};
for (auto it = vec.begin(); it != vec.end(); ++it) {
    cout << *it << " "; // 输出: 1 2 3
}
```
- **类型**：`begin()`/`end()`（正向），`rbegin()`/`rend()`（反向）。
- **范围for循环**（C++11起）：
```cpp
for (int x : vec) {
    cout << x << " ";
}
```

### 7.3 算法
STL算法（`<algorithm>`）操作容器，提供排序、搜索、修改等功能。
- **排序**：
```cpp
sort(vec.begin(), vec.end()); // 升序
sort(vec.begin(), vec.end(), greater<int>()); // 降序
```
- **搜索**：
```cpp
auto it = find(vec.begin(), vec.end(), 2); // 查找2
if (it != vec.end()) cout << "Found: " << *it;
```
- **其他**：
  - `min_element`, `max_element`：找最小/最大值。
  - `accumulate`（`<numeric>`）：求和。
  ```cpp
  #include <numeric>
  int sum = accumulate(vec.begin(), vec.end(), 0); // 求和
  ```

### 7.4 自定义比较器
自定义排序规则：
```cpp
struct Compare {
    bool operator()(int a, int b) const { return a > b; }
};
sort(vec.begin(), vec.end(), Compare()); // 降序
```

### 7.5 性能与选择
- `vector`：适合随机访问和尾部操作。
- `list`：适合频繁插入删除。
- `map`/`set`：适合快速查找和排序。
- `unordered_map`/`unordered_set`：适合高性能哈希查找。

---

## 8. 模板（泛型编程）
```cpp
template <typename T>
T max(T a, T b) { return a > b ? a : b; }
```
- **类模板**：
```cpp
template <typename T>
class Box {
public:
    T value;
    Box(T v) : value(v) {}
};
```

---

## 9. 异常处理
```cpp
try {
    if (x == 0) throw runtime_error("Divide by zero!");
    cout << 10 / x;
} catch (const runtime_error& e) {
    cout << "Error: " << e.what();
}
```

---

## 10. 其他高级特性
- **智能指针**（`<memory>`）：
```cpp
unique_ptr<int> ptr = make_unique<int>(5);
shared_ptr<int> sptr = make_shared<int>(10);
```
- **Lambda表达式**：
```cpp
auto add = [](int a, int b) { return a + b; };
cout << add(3, 4); // 输出7
```
- **多线程**（`<thread>`）：
```cpp
#include <thread>
void task() { cout << "Thread running"; }
thread t(task);
t.join();
```

---

## 11. 最佳实践
- 使用`const`确保数据不可变。
- 优先使用智能指针管理内存。
- 善用STL容器和算法。
- 遵循RAII原则管理资源。
- 避免全局变量，保持模块化。

---

## 12. 学习资源
以下是扩展的学习资源，涵盖书籍、在线平台、工具和社区，适合不同学习阶段。

### 12.1 书籍
- **入门**：
  - 《C++ Primer》（5th Edition, Stanley B. Lippman等）：全面介绍C++11/14特性，适合初学者。
  - 《Programming: Principles and Practice Using C++》（Bjarne Stroustrup）：C++之父的入门教程，注重实践。
- **进阶**：
  - 《Effective C++》（Scott Meyers）：55条实用建议，提升代码质量。
  - 《Modern C++ Design》（Andrei Alexandrescu）：深入探讨模板和泛型编程。
  - 《The C++ Standard Library》（Nicolai M. Josuttis）：STL详细指南。
- **参考**：
  - 《C++ Templates: The Complete Guide》（David Vandevoorde等）：模板编程权威书籍。
  - 《C++ Concurrency in Action》（Anthony Williams）：多线程编程实战。

### 12.2 在线平台
- **CPlusPlus.com**（http://www.cplusplus.com）：标准库参考文档，包含详细API说明。
- **cppreference.com**（https://en.cppreference.com）：权威的C++语言和库参考，支持C++11/14/17/20。
- **LeetCode**（https://leetcode.com）：提供C++算法题，适合练习数据结构和算法。
- **HackerRank**（https://www.hackerrank.com）：C++编程挑战，覆盖基础到高级题目。
- **LearnCpp.com**（https://www.learncpp.com）：免费的结构化教程，适合自学。
- **Coursera/edx**：提供C++课程，如Coursera的“C++ For C Programmers”。

### 12.3 工具
- **编译器**：
  - GCC（GNU Compiler Collection）：免费、跨平台，支持最新C++标准。
  - Clang：性能优异，错误信息友好，适合开发和调试。
  - MSVC（Microsoft Visual C++）：Windows平台优化，支持Visual Studio。
- **IDE**：
  - Visual Studio：功能强大的Windows IDE，支持调试和代码补全。
  - CLion：JetBrains的跨平台C++ IDE，适合大型项目。
  - Code::Blocks：轻量级开源IDE，适合初学者。
  - Visual Studio Code：轻量编辑器，配合C++插件使用。
- **调试与分析**：
  - GDB：命令行调试工具，配合GCC使用。
  - Valgrind：检测内存泄漏和性能问题。
  - CMake：跨平台构建工具，简化项目管理。

### 12.4 社区与论坛
- **Stack Overflow**（https://stackoverflow.com）：C++相关问题解答，搜索历史问题或提问。
- **Reddit**（r/cpp）：C++社区，讨论新特性、工具和最佳实践。
- **GitHub**：搜索C++开源项目（如Boost、Eigen），学习高质量代码。
- **C++ Slack/Discord**：加入C++开发者社区，实时交流。

### 12.5 实践项目
- 实现简单游戏（如贪吃蛇、井字棋）练习OOP和STL。
- 开发小型数据库管理系统，掌握文件I/O和数据结构。
- 贡献C++开源项目，提升实战经验。

### 12.6 标准与更新
- 跟踪C++标准（C++11/14/17/20/23）变化，cppreference.com提供最新特性说明。
- 关注ISO C++委员会（https://isocpp.org）博客，了解语言发展动态。

---