# 20道常考 Python 面试题大总结

一般来说，**面试官会根据求职者在简历中填写的技术及相关细节来出面试题。**一位拿了大厂技术岗Special Offer的同学分享了他总结的面试经验。今天分享给大家：
**·** 数据类型有几种、有什么区别
**·** 进程、线程、协程的定义及区别
**·** 深浅拷贝的区别
**·** 常用开发模式
**·** 函数式编程、对象式编程
**·** 闭包、装饰器
**·** 垃圾回收机制
**·** linux常用命令，举例说明
根据该网友的经验，以上是面试题的常考范围，如果能答出来大部分内容，说明技术水平基本没太大问题。**建议每个问题至少答三点，同时注意观察面试官的反应**，如果觉得面试官感兴趣的话可以多说一些，不感兴趣的话则可适当地少说。**平均每个问题回答控制在3-5分钟比较合适。**
**技术问题一般会问15个左右，一轮面试的时长基本在一小时以上。**一小时以下的面试成功希望可能会小一些。所以，建议大家在技术基础方面一定要准备充分、多下功夫。

## 20道常考Python面试题
我们为大家精心奉上Python面试宝典中最常考的20道面试题。看看你都会做么？

### 1、如何在Python中管理内存？
Python中的内存管理由Python私有堆空间管理。对象和数据结构位于私有堆中，开发者无权访问此私有堆，是Python解释器负责处理的。Python对象的堆空间分配由内存管理器完成。核心API提供了一些开发者编写代码的工具。Python内置的垃圾回收器会回收使用所有的未使用内存，使其适用于堆空间。

### 2、解释Python中的Help()函数和Dir()函数。
Help()函数是一个内置函数，作用是查看函数和详细说明模块用途。
![640](./img/v3smLKjiqrihVCci/1722921561208-4f31d782-4600-4524-b5d3-0adb62f5ae94-230548.png)
运行结果是：
![640](./img/v3smLKjiqrihVCci/1722921561285-eb672e94-9208-45ee-9dc5-777cb73145af-152676.png)
Dir()函数是Python内置函数，Dir() 函数不带参数时，返回当前范围内的变量、方法和定义的类型列表；带参数时，返回参数的属性、方法列表。
举个例子展示其使用方法：
![640](./img/v3smLKjiqrihVCci/1722921561389-d7a8c635-efb8-411a-b375-d847c109093a-829222.png)
运行结果是：
![640](./img/v3smLKjiqrihVCci/1722921561307-ecb99ba9-5a15-4c9d-82ab-dc7a4ccb91f9-271861.png)

### 3、当Python退出时，是否会清除所有分配的内存？
答案是否。当Python退出时，对其他对象具有循环引用的Python模块，以及从全局名称空间引用的对象不会被解除分配或释放。无法解除分配C库保留的那些内存部分。退出时，由于拥有自己的高效清理机制，Python会尝试取消分配/销毁其他所有对象。

### 4、什么是猴子补丁？
在运行期间动态修改一个类或模块。
![640](./img/v3smLKjiqrihVCci/1722921561343-47e682f5-258a-475a-b6fa-016054ceeede-632292.png)
运行结果是：
![640](./img/v3smLKjiqrihVCci/1722921561306-ce10fd45-d1c7-42f1-b31d-3bb166253303-783841.png)

### 5、Python中的字典是什么？
字典指的是Python中的内置数据类型。它定义了键和值之间的一对一关系，包含了一对键及其对应的值。字典由键索引。

### 6、解释一下Python中的逻辑运算符。
Python中有3个逻辑运算符：and，or，not。

### 7、为什么不建议以下划线作为标识符的开头？
Python没有私有变量的概念，所以约定速成以下划线为开头来声明一个变量为私有。如果不想让变量私有，则不要使用下划线开头。

### 8、什么是Flask？
Flask是Python编写的一款轻量级Web应用框架。WSGI 工具箱采用 Werkzeug ，模板引擎使用 Jinja2。Flask使用 BSD 授权。Werkzeug和Jinja2是其中的两个环境依赖。Flask不需要依赖外部库。

### 9、解释Python中的join()和split()函数。
Join()可用于将指定字符添加至字符串中。
![640](./img/v3smLKjiqrihVCci/1722921561338-efc80ca8-af52-4c6a-b02a-049faa3fc267-854104.png)
运行结果是：
![640](./img/v3smLKjiqrihVCci/1722921561294-7a58172b-1755-4da2-8e8e-5822638be04c-988678.png)
Split()可用于指定字符分割字符串。
![640](./img/v3smLKjiqrihVCci/1722921561377-f076dae9-5110-4320-acf8-4f3f21382f65-365080.png)
运行结果是：
![640](./img/v3smLKjiqrihVCci/1722921561328-42f7f2ed-0aae-4120-b374-e3f591891b28-366554.png)

### 10、Python中的标识符长度有多长？
标识符可以是任意长度。在命名标识符时还必须遵守以下规则：
**·** 只能以下划线或者 A-Z/a-z 中的字母开头
**·** 其余部分可以使用 A-Z/a-z/0-9
**·** 区分大小写
**·** 关键字不能作为标识符

### 11、Python中是否需要缩进？
需要。Python指定了一个代码块。循环，类，函数等中的所有代码都在缩进块中指定。通常使用四个空格字符来完成。如果开发者的代码没有缩进，Python将无法准确执行并且也会抛出错误。

### 12、请解释使用*args的含义。
当我们不知道向函数传递多少参数时，比如我们向传递一个列表或元组，我们就使用*args。
![640](./img/v3smLKjiqrihVCci/1722921561401-11509003-7e23-418f-9064-41382214f2ca-336462.png)
运行结果是：
![640](./img/v3smLKjiqrihVCci/1722921561436-4c8078b7-50b6-411f-b5f5-d28ab068bc7d-509168.png)

### 13、深拷贝和浅拷贝之间的区别是什么？
浅拷贝是将一个对象的引用拷贝到另一个对象上，如果在拷贝中改动，会影响到原对象。深拷贝是将一个对象拷贝到另一个对象中，如果对一个对象的拷贝做出改变时，不会影响原对象。

### 14、Python中如何实现多线程？
Python是多线程语言，其内置有多线程工具包。多线程能让我们一次执行多个线程。Python中的GIL（全局解释器锁）确保一次执行单个线程。一个线程保存GIL并在将其传递给下个线程之前执行一些操作，看上去像并行运行的错觉。事实上是线程在CPU上轮流运行。所有的传递会增加程序执行的内存压力。

### 15、Python中的闭包是什么？
当一个嵌套函数在其外部区域引用了一个值时，该嵌套函数就是一个闭包。其意义就是会记录这个值。
比如：
![640](./img/v3smLKjiqrihVCci/1722921561446-fc8dcfdf-29c6-40a7-a0cf-c08b8304fc52-736975.png)
运行结果是：
![640](./img/v3smLKjiqrihVCci/1722921561438-011d2c38-44e8-408e-a826-ec456b434236-074373.png)

### 16、Python的优势有哪些？
**·** Python 易于学习
**·** 完全支持面向对象
**·** 高效的高级数据结构，可用少量代码构建出多种功能
**·** 拥有最成熟的程序包资源库之一
**·** 跨平台而且开源

### 17、什么是元组的解封装？
首先，我们先展示解封装：
![640](./img/v3smLKjiqrihVCci/1722921561439-484a9c4f-46fe-4c68-873d-79d051abd9c5-360702.png)
将 3，4，5 封装到元组 mytuple 中，再将值解封装到变量 x，y，z 中：
![640](./img/v3smLKjiqrihVCci/1722921561363-dd851d00-c5e6-4591-b38a-97017cb32720-231718.png)
得到结果为12。

### 18、什么是PEP？
PEP代表Python Enhancement Proposal，是一组规则，指定如何格式化Python代码以获得最大可读性。

### 19、列表和元组之间的区别是什么？
主要区别是列表是可变的，元组是不可变的。比如以下举例：
![640](./img/v3smLKjiqrihVCci/1722921561422-76e67714-3602-41dc-98b5-e2d4e66c4190-113508.png)
会出现以下报错：
![640](./img/v3smLKjiqrihVCci/1722921561423-8cdcccb6-b48a-4a37-b977-ada8cfed1891-561216.png)

### 20、什么是Python模块？Python中有哪些常用的内置模块？
Python模块是包含Python代码的.py文件。此代码可以是函数类或变量。常用的内置模块包括：random、data time、JSON、sys、math等。

# 详解 Python 类的封装、继承和多态！

Python是一种功能强大的编程语言，它支持面向对象编程（OOP），其中类是OOP的核心概念之一。类是一种用户自定义的数据结构，它可以包含属性和方法。在本文中，我们将探讨Python类的基本概念、用法和一些示例。
首先，让我们来了解一下什么是类。类是一种模板或蓝图，用于创建对象。对象是类的实例，它具有类定义的属性和方法。类定义了对象的行为和状态。通过类，我们可以创建多个相似的对象，每个对象都可以具有不同的属性和方法。

## 创建类
在Python中，我们使用关键字`class`来定义一个类。**「类的定义通常包含属性和方法。属性是类的特征，而方法是类的行为。属性可以是变量，而方法可以是函数。」**下面是一个简单的类的例子：
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def say_hello(self):
        print("Hello, my name is", self.name)

person1 = Person("Alice", 25)
person1.say_hello()
```

在上面的例子中，我们定义了一个名为`Person`的类。它有两个属性`name`和`age`，以及一个方法`say_hello`。`__init__`是一个特殊的方法，用于初始化对象的属性。在创建对象时，我们可以传递参数来设置对象的属性。通过调用`say_hello`方法，对象可以执行特定的行为。

## 使用类
类的属性可以通过对象进行访问和修改。我们可以使用点号（`.`）来访问对象的属性。例如，`person1.name`可以用来获取`person1`对象的`name`属性的值。我们还可以使用点号来修改属性的值，例如`person1.age = 30`可以将`person1`对象的`age`属性的值修改为30。
除了属性，类还可以有其他方法。方法可以接受参数，并且可以在方法内部使用这些参数来执行特定的操作。方法可以返回一个值，也可以不返回任何值。下面是一个例子：
```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def calculate_area(self):
        return self.width * self.height

rectangle1 = Rectangle(5, 3)
area = rectangle1.calculate_area()
print("The area of the rectangle is", area)
```

在上面的例子中，我们定义了一个名为`Rectangle`的类。它有两个属性`width`和`height`，以及一个方法`calculate_area`，用于计算矩形的面积。通过调用`calculate_area`方法，我们可以获取矩形对象的面积。

## 类的继承
Python中的类继承是一种重要的概念，它允许我们创建一个新类，该类继承了另一个类的属性和方法。被继承的类称为父类，新创建的类称为子类。子类可以添加新的属性和方法，或者重写父类的方法。类继承提供了代码重用的机制，可以减少代码的重复编写，并提高代码的可读性和可维护性。
让我们通过一个例子来理解类继承的概念。假设我们有一个父类`Animal`，它有一个属性`name`和一个方法`make_sound`，用于描述动物的名字和发出的声音。我们可以定义一个子类`Dog`，它继承了`Animal`类的属性和方法，并且可以添加新的属性和方法。
```python
class Animal:
    def __init__(self, name):
        self.name = name

    def make_sound(self):
        print("The animal makes a sound")

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed

    def make_sound(self):
        print("The dog barks")

    def fetch(self):
        print("The dog fetches a ball")

animal1 = Animal("Animal")
animal1.make_sound()

dog1 = Dog("Dog", "Labrador")
dog1.make_sound()
dog1.fetch()
```
在上面的例子中，我们定义了一个名为`Animal`的父类，它有一个属性`name`和一个方法`make_sound`。然后，我们定义了一个名为`Dog`的子类，它继承了`Animal`类的属性和方法，并添加了一个新的属性`breed`和一个新的方法`fetch`。通过调用`make_sound`方法，我们可以看到不同的输出。还可以通过调用`fetch`方法来查看狗的特定行为。

在子类中，我们可以使用`super().__init__(name)`来调用父类的`__init__`方法，并传递参数`name`。这样，子类就可以继承父类的属性。我们还可以重写父类的方法，如在`Dog`类中重写了`make_sound`方法。

除了单继承，Python还支持多继承，即一个子类可以继承多个父类的属性和方法。使用逗号分隔父类的名称即可实现多继承。例如：
```python
class Parent1:
    def method1(self):
        print("This is method 1")

class Parent2:
    def method2(self):
        print("This is method 2")

class Child(Parent1, Parent2):
    def method3(self):
        print("This is method 3")

child = Child()
child.method1()
child.method2()
child.method3()
```

在上面的例子中，我们定义了两个父类`Parent1`和`Parent2`，它们分别有自己的方法。然后，我们定义了一个子类`Child`，它继承了`Parent1`和`Parent2`的属性和方法。通过创建子类的对象，我们可以调用父类的方法。

需要注意的是，当一个子类继承了多个父类时，如果多个父类有相同的方法名，子类将继承第一个遇到的方法。如果需要调用其他父类的方法，可以使用父类的名称加上方法名来调用。

## 类的封装
封装是面向对象编程中的一个重要概念，它允许我们将数据和方法包装在一个类中，并通过访问控制来保护数据的安全性。通过封装，我们可以隐藏类的内部实现细节，只暴露必要的接口给外部使用。这样可以提高代码的可维护性和可扩展性，并且减少了代码的耦合性。
让我们通过一个例子来理解类封装的概念。假设我们有一个类`Person`，它有两个属性`name`和`age`，以及两个方法`get_name`和`get_age`，用于获取属性的值。我们可以将属性设置为私有的，以防止外部直接访问和修改。
```
class Person:
    def __init__(self, name, age):
        self.__name = name
        self.__age = age

    def get_name(self):
        return self.__name

    def get_age(self):
        return self.__age

person1 = Person("John", 25)
print(person1.get_name())
print(person1.get_age())
```
在上面的例子中，我们使用双下划线`__`将属性`name`和`age`设置为私有属性。这样，外部无法直接访问和修改这些属性。我们通过定义`get_name`和`get_age`方法来获取属性的值，并将其作为类的接口暴露给外部。
通过封装，我们可以确保属性的安全性，只允许通过类的方法来访问和修改属性。这样，我们可以在方法中添加一些验证逻辑，以确保属性的有效性。同时，如果需要修改属性的实现细节，我们只需要修改类的内部代码，而不会影响到外部使用类的代码。
除了私有属性，我们还可以使用装饰器`@property`来定义属性的访问器和修改器方法，以提供更灵活的属性访问方式。例如：
```python
class Person:
    def __init__(self, name, age):
        self.__name = name
        self.__age = age

    @property
    def name(self):
        return self.__name

    @property
    def age(self):
        return self.__age

    @age.setter
    def age(self, value):
        if value < 0:
            raise ValueError("Age cannot be negative")
        self.__age = value

person1 = Person("John", 25)
print(person1.name)
print(person1.age)

person1.age = 30
print(person1.age)

person1.age = -5 # Raises an exception
```
在上面的例子中，我们使用`@property`装饰器来定义`name`和`age`属性的访问器方法。这样，我们可以通过`person1.name`和`person1.age`的方式来获取属性的值，就像访问普通的属性一样。同时，我们还使用`@age.setter`装饰器来定义`age`属性的修改器方法，以实现对属性的赋值操作。在修改器方法中，我们可以添加一些验证逻辑，以确保属性的有效性。Python中的类封装是一种重要的概念，它允许我们将数据和方法包装在一个类中，并通过访问控制来保护数据的安全性。通过封装，我们可以隐藏类的内部实现细节，只暴露必要的接口给外部使用。通过私有属性和装饰器`@property`，我们可以实现更灵活的属性访问方式。

## 类的多态
多态是面向对象编程中的一个重要概念，它允许我们使用相同的接口来处理不同的对象类型。通过多态，我们可以编写通用的代码，使其能够适应不同的对象类型，提高代码的灵活性和可复用性。
让我们通过一个例子来理解多态的概念。假设我们有一个基类`Animal`，它有一个方法`make_sound`，用于发出动物的叫声。然后我们派生出两个子类`Dog`和`Cat`，它们分别重写了`make_sound`方法。
```python
class Animal:
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        print("Woof!")

class Cat(Animal):
    def make_sound(self):
        print("Meow!")

def animal_sound(animal):
    animal.make_sound()

dog = Dog()
cat = Cat()

animal_sound(dog) # 输出 "Woof!"
animal_sound(cat) # 输出 "Meow!"
```

在上面的例子中，我们定义了一个函数`animal_sound`，它接受一个`Animal`类型的参数，并调用其`make_sound`方法。在调用`animal_sound`函数时，我们传入了一个`Dog`对象和一个`Cat`对象。由于`Dog`和`Cat`都是`Animal`的子类，它们都实现了`make_sound`方法，因此在调用`animal_sound`函数时，会根据对象的实际类型来调用相应的方法。
这就是多态的实现方式，通过基类的接口来处理不同的子类对象，实现了代码的通用性和灵活性。在上面的例子中，我们可以轻松地添加更多的子类，例如`Bird`、`Cow`等，只需要实现它们自己的`make_sound`方法，并调用`animal_sound`函数即可。
除了通过继承实现多态，Python还提供了另一种实现多态的方式，即通过鸭子类型（Duck Typing）。鸭子类型是一种动态类型判断方式，它关注对象的行为而不是类型。只要一个对象具有特定的方法或属性，就可以被视为具有某个类型的行为。
```python
class Duck:
    def quack(self):
        print("Quack!")

class Person:
    def quack(self):
        print("I'm quacking like a duck!")

def make_quack(obj):
    obj.quack()

duck = Duck()
person = Person()

make_quack(duck)   # 输出 "Quack!"
make_quack(person) # 输出 "I'm quacking like a duck!"
```
在上面的例子中，我们定义了一个函数`make_quack`，它接受一个对象作为参数，并调用其`quack`方法。在调用`make_quack`函数时，我们传入了一个`Duck`对象和一个`Person`对象。尽管`Duck`和`Person`是不同的类，但它们都具有`quack`方法，因此可以被视为具有相同类型的行为。Python中的多态允许我们使用相同的接口来处理不同的对象类型，提高了代码的灵活性和可复用性。通过继承和鸭子类型，我们可以实现多态。

## 类的魔术方法
Python中的魔术方法（Magic Methods）是一组特殊的方法，它们以双下划线开头和结尾，用于定义和控制类的行为。这些方法被称为魔术方法，因为它们在特定的情况下会自动被调用，而不需要我们显式地调用它们。
下面是一些常用的魔术方法及其功能：

1.  `__init__`: 初始化方法，在创建对象时自动调用。 
2.  `__str__`: 返回对象的字符串表示，可以通过`str(obj)`或`print(obj)`调用。 
3.  `__repr__`: 返回对象的字符串表示，可以通过`repr(obj)`调用。 
4.  `__len__`: 返回对象的长度，可以通过`len(obj)`调用。 
5.  `__getitem__`和`__setitem__`: 实现对象的索引访问，可以通过`obj[key]`访问和修改元素。 
6.  `__iter__`和`__next__`: 实现对象的迭代，可以使用`for`循环遍历对象。 
7.  `__call__`: 将对象作为函数调用，可以使用`obj()`调用对象。 

下面是一个简单的示例，演示了如何使用魔术方法来自定义一个简单的向量类：
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        else:
            raise TypeError("Unsupported operand type")

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        else:
            raise TypeError("Unsupported operand type")

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.x == other.x and self.y == other.y
        else:
            return False

v1 = Vector(1, 2)
v2 = Vector(3, 4)

print(v1)         # 输出 "Vector(1, 2)"
print(v1 + v2)    # 输出 "Vector(4, 6)"
print(v1 * 2)     # 输出 "Vector(2, 4)"
print(v1 == v2)   # 输出 "False"
```
在上面的例子中，我们定义了一个向量类`Vector`，并实现了`__init__`、`__str__`、`__add__`、`__mul__`和`__eq__`等魔术方法。通过这些魔术方法，我们可以自定义向量类的初始化、字符串表示、加法、乘法和相等性等行为。
通过使用魔术方法，我们可以使类的行为更加符合我们的预期，并且可以与Python内置的函数和操作符进行交互。这使得我们可以更好地控制和定制我们的类的行为。

## 总结
总结一下，Python类是一种用户自定义的数据结构，它包含属性和方法。类是对象的模板，通过类可以创建多个相似的对象。类可以继承其他类的属性和方法，并且可以重写父类的方法。类还可以定义一些特殊的方法，用于实现类的特殊行为。通过使用类，我们可以更好地组织和管理代码，提高代码的可读性和重用性。希望本文对你理解Python类有所帮助！


> 原文: <https://www.yuque.com/lucky-bk3s1/sc1v5b/rba414dgfvwbp01t>








> 原文: <https://www.yuque.com/lucky-bk3s1/sc1v5b/igo3codyi6513ddu>