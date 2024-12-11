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

## Position Doesn't Matter When template specilaztion
In C++, the order of template specializations generally doesn’t matter as long as the compiler can match the correct specialization based on the given arguments.
 ```c++
  template <class Tuple, class Elem, class Enable = void>
  struct has_elem : std::false_type {};
  template <class Elem>
  struct has_elem<Elem, Elem> : std::true_type {};
  // The first function
  template <class Tuple, class Elem>
  struct has_elem<Tuple, Elem, std::enable_if_t<std::tuple_size<Tuple>::value> >
      : has_elem<Tuple, Elem, std::make_index_sequence<std::tuple_size<Tuple>::value> > {};
  // The second function
  template <class Tuple, class Elem, int... Is>
  struct has_elem<Tuple, Elem, std::index_sequence<Is...>>
      : std::disjunction<has_elem<std::tuple_element_t<Is, Tuple>, Elem>...> {};
 ```
In C++, templates don't just inherit in the way regular classes do, but they can delegate responsibilities and inherit behaviors via template specialization.
The first function: 1: It uses std::enable_if_t<std::tuple_size<Tuple>::value> to ensure that it only gets instantiated for valid tuple types (non-empty tuples). 2: It then invokes the first template by generating an index sequence via std::make_index_sequence<std::tuple_size<Tuple>::value> and passing it along to the second template.
The key idea here is that the first template doesn’t exactly "inherit" from the second in the traditional C++ inheritance sense, but instead it specializes the has_elem structure for a particular case (when the tuple is non-empty), and delegates the recursive checking behavior to the second template.

