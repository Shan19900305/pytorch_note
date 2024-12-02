# Basic Concept
从0开始学习基本概率

## Hierarchy Layout
Hierarchy layout相对torch中的size/stride进行了延伸,其支持嵌套结构的封装。论文中的说明:We introduce a novel representation for tensor shapes, layouts and tiles. Graphene's tensors are decomposable into tiles represented as smaller nested tensors。表示方式为:((内部行数,外部行数1, 外部行数2,...),(内部列数,外部列数1,外部列数2,...))。
 - 例子1:Hierarchy layout -> Normal layout
   假设Hierarchy layout的shape为:((2,4), (3,5)),stride为:((1, 6), (2,24))。其表示内层数据块的shape为(2,3),且为行主序。外层数据块的shape为(4,5),且为列主序。
   <img src="Hierarchy layout to normal layout.png" width="400" height="300">
   转化为Normal Layout,其shape为:(4,5,2,3),stride为:(6,24,3,1)。
   对应取数坐标如下:
   auto row_coord = make_coord(1, 2); 
   auto col_coord = make_coord(2, 1); 
   auto coord = make_coord(row_coord, col_coord);
   则对应外层数据块坐标为(2,1),内层数据块坐标为(1,2),也就是数值为41的坐标。转换为普通坐标可以认为是(2, 1, 1, 2),根据公式:2 * 6 + 24 + 3 + 2 * 1 = 41。

 - 例子2:Normal layout -> Hierarchy layout
  假设torch的基本Tensor shape为(B,M,K),stride为(M * K,K,1),其表示为B个M*K大小的矩阵,其内部元素为K个连续的元素,外部元素为M*K个连续的元素。
  转化为Hierarchy layout的表示范围为:shape为(M, (K, B)),stride为(K,(1, M * K)),也就是内层数据块为shape(M, K),stride为(K, 1);外层数据块的shape为(1, B),stride为(1,M * K)。

## Tuple对象实现
Layout实现的基本数据结构,相对与标准库tuple实现的更加简洁。特别地对于is_empty<T>为true的类型对象,不进行真实的构造存储,仅在get函数时进行构造返回。
  ```mermaid
    graph LR
    A[struct tuple]  -->|inherit| B[struct TupleBase] -->|inherit| C[struct EBO]
  ```
### 基础实现:
  - 基础存储单元EBO: 1）当IsEmpty为true时,不进行任何类型实例化,仅表示一个模板的特化类型; 2）当IsEmpty是false时,会真实地进行对象T实例化和存储。
    ```mermaid
      graph LR
      A[struct EBO with template]  --> C[value = is_empty<T>::value>]
        C -->|value == true| D[Only represents a specialized type of a template, don't do any instantiation.] --> F
        C -->|value == false| E[init a value of type T, and stored in class.]  --> F[getv function with IsEmpty]
        F -->|IsEmpty == true| G[return T instantiation]
        F -->|IsEmpty == false| H[return t.t_]
    ```
    ```c++
      template <size_t N, class T, bool IsEmpty = is_empty<T>::value>
      struct EBO;
    ```
  - 转接TupleBase:其主要作用就是将参数模板展开,转接给各个EBO特化类。
    ```c++
      template <class IdxSeq, class... T>
      struct TupleBase;

      // Base class of cute::tuple binds each element to an index
      // by inheriting from EBO<i, t> for each (i, t) in (I..., T...).
      // The storage (for nonempty t) lives in the base classes.
      template <size_t... I, class... T>
      struct TupleBase<index_sequence<I...>, T...>
          : EBO<I,T>...
    ```
  - tuple:面向用户侧使用,定义了各种处理方法。
    ```c++
      template <class... T>
      struct tuple : detail::TupleBase<make_index_sequence<sizeof...(T)>, T...>
    ```

### 基本使用方式和运算方法
 - rank: 获取对应维度元素个数,或者总的元素个数,其中嵌套结构算1个元素。
  - example:
     tuple<tuple<int, _2>, int, int>>tuple_3h_m(tuple<int,_2>(1,_2{}), 8, 2);
     tuple<int, int, _2> tuple_3d_m(8,4,_2{});
     rank(tuple_3h_m);     // Int<3>{}
     rank<0>(tuple_3h_m);  // Int<2>{}
     rank(tuple_3d_m);     // Int<3>{}
  -code:
   ```c++
     template <int... Is, class IntTuple>
     CUTE_HOST_DEVICE constexpr
     auto
     rank(IntTuple const& t)
     {
       if constexpr (sizeof...(Is) == 0) {
         if constexpr (is_tuple<IntTuple>::value) {
           return Int<tuple_size<IntTuple>::value>{};
         } else {
           return Int<1>{};
         }
       } else {
         return rank(get<Is...>(t));
       }
       CUTE_GCC_UNREACHABLE;
     }
   ```

 - product: 进行求内积的处理,使用折叠表达式进行运算处理。如果存在嵌套结构,则嵌套调用Product进行处理。
            非tuple时,返回Int<1>{}。
 - product_each: 对多个Tuple分别进行求内积。
   - product example1:
    tuple<tuple<int, _2>, int, int> tuple_3h_m(tuple<int,_2>(1,_2{}),8,2);
    result: product(tuple_3h_m)  ->  32
   - product example2:
    tuple<tuple<_1, _2>, _3, _2> tuple_3h_m;
    result:product(tuple_3h_m)  ->  _12{} // 类型不发生改变
 - product_like:按照右操作数的元素个数,对左操作数进行求内积,类似理解按照归约操作。(可以简单理解为将tuple的子序列规约为单个数值。)
   - product_like example1:
     tuple<tuple<int, _2>, int, int>>tuple_3h_m(tuple<int,_2>(1,_2{}), 8, 2);
     tuple<int, int, _2> tuple_3d_m(8,4,_2{});
     auto result = product_like(tuple_3h_m, tuple_3d_m); // tuple<int, int, int> result{2, 8, 2};
   - code:
    ```c++
      struct Product
      {
        template <class IntTuple>
        CUTE_HOST_DEVICE constexpr
        auto
        operator()(IntTuple const& a) const
        {
          if constexpr (is_tuple<IntTuple>::value) {
            if constexpr (tuple_size<IntTuple>::value == 0) {
              return Int<1>{};
            } else {
              // Product{} is a functor and used when a is nested structure.
              return cute::transform_apply(a, Product{}, multiplies_unary_lfold{});
            }
          } else if constexpr (cute::is_integral<IntTuple>::value) {
            return a;
          }

          CUTE_GCC_UNREACHABLE;
        }
      };
      // Callable product function object
      CUTE_INLINE_CONSTANT Product product;
    ```
 - max

### 基本运算规则
 - coalesce:
   


probShape是实际输入的矩阵,tileShape是拆分维度信息,对gemm来说,tileShape这个是一个cluster处理的数据,刚好拆分给4个ipu。对unary来说,还要继续拆atomShape

template <class ElementwiseUnaryOperation,
          class ElementwiseUnaryLayout_ = Layout<Shape<_1, _1>>,
          class BlockTiler_     = Tile<Underscore, Underscore>,
          class AtomTiler_     = Tile<Underscore, Underscore>>
struct TiledElementwiseUnary : ElementwiseUnary_Traits<ElementwiseUnaryOperation> {}
参数说明:
ElementwiseUnaryOperation:操作函数,如log,exp,sigmoid等。
ElementwiseUnaryLayout_:表示block中各个Thread的排列方式,对应到MLU就是各个ipu的排列方式,默认为列主序;
BlockTiler_:表示总的处理数据量;
AtomTiler_:表示block中每个Thread处理的数据量,即AtomTiler按照ElementwiseUnaryLayout_排列,就是单次单个cluster处理的数据量。


TiledElementwiseUnary负责进行数据拆分,然后调用ElementwiseUnaryOperation进行计算。



论文链接: https://dl.acm.org/doi/pdf/10.1145/3582016.3582018
参考资料: https://zhuanlan.zhihu.com/p/661182311
参考资料: https://zhuanlan.zhihu.com/p/662089556

# 辅助工具及方法
## 单测试用例执行方法



# C++知识点
## is_empty
If T is an empty type (that is, **a non-union class type with no non-static data members other than bit-fields of size 0, no virtual functions, no virtual base classes, and no non-empty base classes**), provides the member constant value equal to true. For any other type, value is false.
 - example:
   ```c++
   std::cout << std::is_empty<int>::value << std::endl; // false
   struct Test1 { };
   struct Test2 { int a; };
   struct Test3 { static int a; };
   std::cout << std::is_empty<Test1>::value << std::endl; // true
   std::cout << std::is_empty<Test2>::value << std::endl; // false
   std::cout << std::is_empty<Test3>::value << std::endl; // true
   ```

## constexpr functions
A constexpr function allows its return value to be computed at compile time if the function arguments are constant expressions.
 - A constexpr function or constructor is implicitly inline;
 - A constexpr function must accept and return only literal types;
 - A constexpr function can be recursive
 - Before C++20, a constexpr function can't be virtual, and a constructor can't be defined as   constexpr when the enclosing class has any virtual base classes. In C++20 and later, a constexpr function can be virtual;
 - The body can be defined as = default or = delete;
 - The body can contain no goto statements or try blocks;
 - An explicit specialization of a non-constexpr template can be declared as constexpr;
 - An explicit specialization of a constexpr template doesn't also have to be constexpr.
 - example:
   ```c++
      constexpr int square(int x) {
          return x * x;
      }

      int main() {
          constexpr int result = square(5);  // Compile-time computation
          static_assert(result == 25, "Result must be 25");

          std::cout << "Square of 5: " << result << std::endl;  // Runtime output
          return 0;
      }
   ```

## constexpr constructor
A constexpr constructor allows the construction of objects at compile time.(Form GPT-4O)
 - The constructor must initialize all non-static member variables either in the member initializer list or as in-class initializers.
 - The constructor body must consist of valid constant expressions (e.g., no dynamic memory allocation, no non-constexpr function calls, etc.).
 - The class must have all its member variables and base classes (if any) also constexpr-compatible.
 - a constexpr constructor can be executed at runtime under certain conditions. While constexpr constructors are designed to enable compile-time initialization, they can also be used for runtime initialization if the object they are initializing does not meet the requirements for a compile-time constant
 - example:
    ```c++
      #include <iostream>

      class Point {
      public:
          constexpr Point(double x, double y) : x_(x), y_(y) {}

          constexpr double x() const { return x_; }
          constexpr double y() const { return y_; }
          constexpr ~Point() {}  // C++20 allows default destructor

      private:
          double x_;
          double y_;
      };

      int main() {
          // Create a constexpr object at compile time
          constexpr Point p(3.0, 4.0);
          int value = 3.0;
          // constexpr Point p(value, 4.0); read of non-const variable 'x' is not allowed in a constant expression
          Point p(value, 4.0); // this is ok and runtime initialization.
          auto* ptr = new Point(3.0, 4.0); // this is ok and runtime initialization.
          delete ptr;

          // Access members at compile time
          static_assert(p.x() == 3.0, "X coordinate must be 3.0");
          static_assert(p.y() == 4.0, "Y coordinate must be 4.0");

          // Access members at runtime
          std::cout << "Point: (" << p.x() << ", " << p.y() << ")" << std::endl;

          return 0;
      }
    ```
参考链接:
https://learn.microsoft.com/en-us/cpp/cpp/constexpr-cpp?view=msvc-170
https://en.cppreference.com/w/cpp/language/constexpr

## base class lookup
取自GPT4o: base class lookup in C++, it means the compiler will examine the inheritance chain and the member declarations in the base classes to determine which one to use
https://isocpp.org/wiki/faq/strange-inheritance
 - example:
    ```c++
      #include <iostream>
      #include <utility>

      template <size_t N, class T>
      struct EBO
      {
        constexpr EBO() : t_{} {}
        constexpr EBO(T const& t) : t_{t} {}
        T t_;
      };

      template <class IdxSeq, class... T>
      struct TupleBase;
      template <size_t... I, class... T>
      struct TupleBase<std::index_sequence<I...>, T...> : EBO<I, T>... {
        constexpr TupleBase() {}
        constexpr TupleBase(T const&... t) : EBO<I,T>(t)... {}
      };


      template<typename... T>
      struct Tuple : public TupleBase<std::make_index_sequence<sizeof...(T)>, T...> {
        constexpr Tuple() {}
        constexpr Tuple(T... t) : TupleBase<std::make_index_sequence<sizeof...(T)>, T...>(t...) {}
      };

      template<size_t N, typename T>
      constexpr auto getv(EBO<N, T> const& a) {
        return a.t_;
      }

      template<size_t N, typename... T>
      constexpr auto get(Tuple<T...> const& a) {
        return getv<N>(a);
      }

      int main() {
        // constexpr TupleBase<std::index_sequence<1, 2, 3, 4>, int, int, int, int> a;
        constexpr Tuple<int,int,int> a_a(1,23,4);
        std::cout << get<2>(a_a) << std::endl;  // the compiler will automatically perform base class lookup to find the matching base class during the getv call
        return 0;
      }
    ```


## Assignment operator for base and derived class.
取自:https://www.quora.com/What-will-happen-in-C-if-the-base-class-has-a-defined-assignment-operator-but-the-derived-class-does-not
In C++, if the base class has a defined assignment operator but the derived class does not explicitly define one, the derived class will automatically inherit the assignment operator from the base class. However, there are important considerations regarding how the assignment operator works in this context:

 - Base Class Assignment Operator: If the base class has a user-defined assignment operator, that operator will be used when assigning one derived class object to another derived class object, but only the base part of the objects will be assigned. The derived part will not be properly handled unless the derived class explicitly defines its own assignment operator.
 - Slicing Problem: If you assign a derived class object to a base class object (or reference), object slicing will occur. This means that only the base class part of the derived class object will be copied, and any additional data members or functionalities in the derived class will be lost.
 - Default Behavior: If the derived class does not have any additional members (i.e., it only inherits from the base class), the inherited assignment operator will work correctly, as it will effectively perform a member-wise assignment using the base class's operator.
 - Virtual Functions: If the base class has virtual functions and you assign a derived class object to a base class object, the virtual mechanism still works when accessing those functions, but the derived part is still sliced off.
 - example:
    ```c++
      class Base { 
        public:
          int baseValue;
          Base& operator=(const Base& other) {
              if (this != &other) {
                  baseValue = other.baseValue;
              }
              return *this;
          }
      };
      class Derived : public Base {
      public:
        int derivedValue; // Additional member
        // No assignment operator defined
      };

      int main() {
        Derived d1;
        d1.baseValue = 1;
        d1.derivedValue = 2;
        Derived d2;
        d2 = d1; // Uses Base's assignment operator
        // After assignment:
        // d2.baseValue will be 1 (copied from d1)
        // d2.derivedValue will be uninitialized (not copied)
      }
    ```
## Fold Expressions（C++17）
Fold expressions are a feature in C++ that allow you to perform operations on a range of elements in a container or array. They are a powerful tool for reducing the amount of code you need to write and make your code more concise and readable.
 - left fold: 从左到右进行参数包处理,((1 + 2) + 3) + 4。
  ```c++
    template <typename... Args>
    auto sum(Args... args) {
      return (args + ...);  // 左折叠表达式
    }
  ```
 - right fold:从右到左进行参数包处理,1 + (2 + (3 + 4))。
  ```c++
    template <typename... Args>
    auto sum(Args... args) {
      return (... + args);  // 右折叠表达式
    }
  ```