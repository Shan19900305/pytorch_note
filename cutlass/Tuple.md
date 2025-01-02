# Tuple对象实现
Layout实现的基本数据结构,相对与标准库tuple实现的更加简洁。特别地对于is_empty<T>为true的类型对象,不进行真实的构造存储,仅在get函数时进行构造返回。
  ```mermaid
    graph LR
    A[struct tuple]  -->|inherit| B[struct TupleBase] -->|inherit| C[struct EBO]
  ```
## 基础实现:
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

## Tuple基本算法实现
### 相关接口
 - Tuple的基本算法实现支持:
   - front, back, take, select, wrap, unwrap, flatten, unflatten, insert, remove, replace, replace_front, replace_back, tuple_repeat, repeat, repeat_like, group, append, prepend,
   iscan, escan, zip, reverse.

### 相关具体实现
  - front: 获取Tuple第一个非Tuple元素，如果为嵌套Tuple，则递归获取。
  - code:
    ```c++
      template <class T>
      CUTE_HOST_DEVICE constexpr
      decltype(auto)
      front(T&& t)
      {
      if constexpr (is_tuple<remove_cvref_t<T>>::value) {
          return front(get<0>(static_cast<T&&>(t)));
      } else {
          return static_cast<T&&>(t);
      }
      CUTE_GCC_UNREACHABLE;
    }
    ```

## Tuple基本运算方法
 - 获取Tuple相关属性: 通过index进行索引获取对应属性，支持多层index索引。
   - 形式： rank<0, 1>(tuple)
   - 具体： rank, shape, depth, size,

 - 获取Tuple元素数值关系： 通过Tuple展开，获取最值等操作，支持多个Tuple处理。
   - 形式： min(tuple1, tuple2)
   - 具体： min, max

 - 获取Tuple元素
   - 形式： 通过Tuple展开，获取求和，乘积等操作，只支持单个Tuple处理。
   - 具体： product, sum, 

 - 二元运算
   - 形式： 两个Tuple对应位置元素进行运算。
   - 具体： inner_product, round_up, shape_min, ceil_div, shape_div, round_up, congruent

### 相关具体实现
 - rank: 通过模板参数获取对应维度元素个数, 或者总的元素个数, 其中嵌套结构算1个元素。
   - Tuple always return size of tuple;
   - Integral dimension always return Int<1>{};
   - Inner tuple rank can using multi-template arguments to get.
   - example:
     ```Plain text
        tuple<tuple<int, _2>, int, int>>tuple_3h_m(tuple<int,_2>(1,_2{}), 8, 2);
        tuple<int, int, _2> tuple_3d_m(8,4,_2{});
        rank(tuple_3h_m);        // Int<3>{}
        rank<0>(tuple_3h_m);     // Int<2>{}
        rank<0, 1>(tuple_3h_m);  // Int<1>{}
        rank(tuple_3d_m);        // Int<3>{}
        rank<1>(tuple_3d_m);     // Integral  dimension Int<1>{}
     ```
   - code:
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

 - shape: 通过模板参数获取对于维度的元素信息，如果对应维度为嵌套的子容器，则返回子容器的shape。
   - Inner tuple rank can using multi-template arguments to get.
   - example：
     ```Plain text
        tuple<tuple<int, _2>, int, int>>tuple_3h_m(tuple<int,_2>(1,_2{}), 8, 2);
        tuple<int, int, _2> tuple_3d_m(8, 4, _2{});
        shape(tuple_3h_m);       // ((1, _2), 8, 2)
        shape<0>(tuple_3h_m);    // (1, _2)
        shape<0, 1>(tuple_3h_m); // 获取嵌套结构中的维度信息 _2
        shape<1>(tuple_3d_m);    // 4
      ```
   - code:
     ```c++
        template <class IntTuple>
        CUTE_HOST_DEVICE constexpr
        auto
        shape(IntTuple const& s)
        {
        if constexpr (is_tuple<IntTuple>::value) {
            return transform(s, [](auto const& a) { return shape(a); });
        } else {
            return s;
        }

        CUTE_GCC_UNREACHABLE;
        }
        // Without copy shape function with index template argument.
     ```

 - depth: 通过模板参数获取对应Tuple的嵌套深度; 默认无嵌套Tuple为1, 非Tuple为0, 而后随着Tuple个数, 逐次增加。
   - Inner tuple depth can using multi-template arguments to get.
   - example：
     ```Plain text
        tuple<tuple<int, _6>, tuple<int, _3>, int> test(tuple<int, _6>(3, _6{}), tuple<int, _3>(4, _3{}), 4);
        depth(tuple_3h_m);       // _2{}
        depth<0>(tuple_3h_m);    // _1{}
        depth<0, 1>(tuple_3h_m); // _0{}
     ```
   - code:
     ```c++
        template <class IntTuple>
        CUTE_HOST_DEVICE constexpr
        auto
        shape(IntTuple const& s)
        {
        if constexpr (is_tuple<IntTuple>::value) {
            return transform(s, [](auto const& a) { return shape(a); });
        } else {
            return s;
        }

        CUTE_GCC_UNREACHABLE;
        }
        // Without copy shape function with index template argument.
     ```
  - size: 根据索引index获取对应嵌套结构的元素个数，内部调用product完成处理。
   - size example：
     ```Plain text
        tuple<tuple<int, _6>, tuple<int, _3>, int> test(tuple<int, _6>(3, _6{}),
                                                        tuple<int, _3>(4, _3{}), 4);
        size(test);       // 864
        size<0>(test);    // 18
        size<0, 1>(test); // _6{}
     ```

 - max / min / gcd: 遍历所有维度元素，然后取最大值 / 最小值 / 最大公约数。其中嵌套结构会被展开铺平。
   - 不支持获取特定嵌套结构或维度的最大值 / 最小值 / 最大公约数，需要通过shape获取对应维度后进行最大值获取。
   - 支持同时传入多个shape，然后取最大值 / 最小值 / 最大公约数。
   - example：
     ```Plain text
        tuple<tuple<int, _6>, int, int> test(tuple<int, _6>(3, _6{}), 9, 4);
        print(max(test));            // 9
        print(max(shape<0>(test)));  // 6
        print(min(test));            // 3
        print(min(shape<0>(test)));  // 3
        print(max(test));            // 1
        print(max(shape<0>(test)));  // 3
     ```
   - code:
     ```c++
        template <class T0, class... Ts>
        CUTE_HOST_DEVICE constexpr
        auto
        max(T0 const& t0, Ts const&... ts)
        {
        if constexpr (is_tuple<T0>::value) {
            return cute::max(cute::apply(t0, [](auto const&... a){ return cute::max(a...); }), ts...);
        } else if constexpr (sizeof...(Ts) == 0) {
            return t0;
        } else {
            return cute::max(t0, cute::max(ts...));
        }

        CUTE_GCC_UNREACHABLE;
        }
     ```

 - product: 进行求内积的处理, 使用折叠表达式进行运算处理。如果存在嵌套结构,则嵌套调用Product进行处理。
            非tuple时, 返回Int<1>{}。
   - product example1:
     ```Plain text
        tuple<tuple<int, _2>, int, int> tuple_3h_m(tuple<int,_2>(1,_2{}),8,2);
        result: product(tuple_3h_m)  ->  32
     ```
   - product example2:
     ```Plain text
        tuple<tuple<_1, _2>, _3, _2> tuple_3h_m;
        result: product(tuple_3h_m)  ->  _12{} // 类型不发生改变
     ```
 - product_each: 对多个Tuple分别进行求内积。
 - product_like: 按照右操作数的元素个数,对左操作数进行求内积,类似理解按照归约操作。
   - 可以简单理解为将tuple中的子tuple序列规约为单个数值或者保持不变。其中会严格对其每个tuple的元素个数，如果元素个数不一致，则编译器会报错。
   - product_like example1:
     ```Plain text
        tuple<tuple<int, _2>, int, int>>tuple_3h_m(tuple<int,_2>(1,_2{}), 8, 2);
        tuple<int, int, _2> tuple_3d_m(8,4,_2{});
        auto result = product_like(tuple_3h_m, tuple_3d_m); // tuple<int, int, int> result{2, 8, 2};
     ```
 - sum: 进行求和的处理，存在嵌套结构时，则会进行嵌套求和。
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

 - inner_product: 二元函数，计算两个Tuple对应元素乘积后的累加和。
   - example：
     ```Plain text
        tuple<tuple<int, _6>, tuple<int, _3>, int> test(tuple<int, _6>(3, _6{}),
                                                        tuple<int, _3>(4, _3{}), 4);
        tuple<tuple<int, _6>, tuple<int, _3>, int> test_right(tuple<int, _6>(2, _6{}),
                                                                tuple<int, _3>(3, _3{}), 2);
        print(inner_product(test, test_right));
     ```
   - code:
     ```c++
        template <class IntTupleA, class IntTupleB>
        CUTE_HOST_DEVICE constexpr
        auto
        inner_product(IntTupleA const& a, IntTupleB const& b)
        {
            if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
            static_assert(tuple_size<IntTupleA>::value == tuple_size<IntTupleB>::value, "Mismatched ranks");
            return transform_apply(a, b, [](auto const& x, auto const& y) { return inner_product(x,y); },
                                        [](auto const&... v) { return (Int<0>{} + ... + v); });
            } else {
            return a * b;
            }

            CUTE_GCC_UNREACHABLE;
        }
     ```

 - ceil_div: 二元函数, 进行维度向上取整的除法操作。
   - 如果左右操作数都为Tuple，则要求左操作数的元素个数要大于等于右操作数。首先进行右操作数维度补齐（尾端补1），然后展开左右操作数进行Integral的除法操作;
   - 如果左操作数是Tuple，右操作数是Integral，首元素与右操作数进行Integral的除法操作;
     - 主要逻辑运算集中在fold的处理，具体如下：
       - 第一个元素除右操作数，取对应运算结果作为结果的第一个元素;
       - 右操作数处以第一个元素，取对应运算结果intermedia;
       - 对应运算结果intermedia和第二个元素进行运算，重复第一个元素的运算步骤，直到对应Tuple或嵌套内层Tuple所有元素都处理完毕。
   - 如果左操作数是Integral，右操作数是Tuple，则先对右操作数进行product操作，然后进行Integral的除法操作;
   - 如果左右操作数都是Integral，则直接进行向上取整的除法操作。
   - example：
     ```Plain text
        tuple<tuple<int, _6>, tuple<int, _3>, int> test(tuple<int, _6>(3, _6{}),
                                                        tuple<int, _3>(4, _3{}), 4);
        tuple<tuple<int, _6>, tuple<_3>, int> test_right(tuple<int, _6>(2, _6{}),
                                                            tuple<_3>(_3{}), 2);
        print(ceil_div(test, 3));           // ((1,6), (4,3), 4)
        print(ceil_div(test, test_right));  // ((2,_1), (2,_3), 2)
    ```
   - code:
     ```c++
        template <class IntTupleA, class IntTupleB>
        CUTE_HOST_DEVICE constexpr
        auto
        ceil_div(IntTupleA const& a, IntTupleB const& b)
        {
            if constexpr (is_tuple<IntTupleA>::value) {
            if constexpr (is_tuple<IntTupleB>::value) {  // tuple tuple
                static_assert(tuple_size<IntTupleA>::value >= tuple_size<IntTupleB>::value, "Mismatched ranks");
                constexpr int R = tuple_size<IntTupleA>::value;        // Missing ranks in TupleB are implicitly 1
                return transform(a, append<R>(b,Int<1>{}), [](auto const& x, auto const& y) { return ceil_div(x,y); });
            } else {                                     // tuple int
                auto const [result, rest] = fold(a, cute::make_tuple(cute::make_tuple(), b),
                [] (auto const& init, auto const& ai) {
                    return cute::make_tuple(append(get<0>(init), ceil_div(ai, get<1>(init))), ceil_div(get<1>(init), ai));
                });
                return result;
            }
            } else
            if constexpr (is_tuple<IntTupleB>::value) {    // int tuple
            return ceil_div(a, product(b));
            } else {
            return (a + b - Int<1>{}) / b;
            }

            CUTE_GCC_UNREACHABLE;
        }
     ```

 - round_up: 二元函数, 对应位置元素进行左操作数对右操作数进行向上取整处理。
   - 仅支持左右操作数都为Tuple或Intiger类型。如果是Tuple且存在嵌套结构，需要满足元素个数一致。如果不存在嵌套结构且元素个数不一致，则进行尾部补1的处理。
   - 元素对齐场景，进行对为的round_up操作。
   - example：
     ```Plain text
        tuple<tuple<int, _6>, tuple<int, _3>, int> test(tuple<int, _6>(3, _6{}),
                                                        tuple<int, _3>(4, _3{}), 4);
        tuple<tuple<int, _6>, tuple<_3>, int> test_right(tuple<int, _6>(2, _6{}),
                                                            tuple<_3>(_3{}), 2);
        tuple<tuple<int>, tuple<int>, int> value{{3}, {2}, 3};
        print(round_up(test, value));       // ((3,6), (4,3), 4)
        print(round_up(test, test_right));  // ((4,_6), (6,_3), 4)
    ```
  - code：
    ```c++
        template <class IntTupleA, class IntTupleB>
        CUTE_HOST_DEVICE constexpr
        auto
        round_up(IntTupleA const& a, IntTupleB const& b)
        {
        if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
            static_assert(tuple_size<IntTupleA>::value >= tuple_size<IntTupleB>::value, "Mismatched ranks");
            constexpr int R = tuple_size<IntTupleA>::value;        // Missing ranks in TupleB are implicitly 1
            return transform(a, append<R>(b,Int<1>{}), [](auto const& x, auto const& y) { return round_up(x,y); });
        } else {
            return ((a + b - Int<1>{}) / b) * b;
        }

        CUTE_GCC_UNREACHABLE;
        }
    ```

 - shape_div: 二元函数, 对应位置元素进行左操作数对右操作数进行向上取整处理。代码处理与ceil_div基本一致。
   - 如果左右操作数都为Tuple，严格要求左右操作数元素个数一致，逐元素进行div运算;
   - 如果左操作数是Tuple，右操作数是Integral，首元素与右操作数进行Integral的除法操作;
     - 主要逻辑运算集中在fold的处理，具体如下：
       - 第一个元素除右操作数，取对应运算结果作为结果的第一个元素;
       - 右操作数处以第一个元素，取对应运算结果intermedia;
       - 对应运算结果intermedia和第二个元素进行运算，重复第一个元素的运算步骤，直到对应Tuple或嵌套内层Tuple所有元素都处理完毕。
   - 如果左操作数是Integral，右操作数是Tuple，则先对右操作数进行product操作，然后进行Integral的除法操作;
   - 如果左右操作数都是Integral，则直接进行Integral的除法操作。
   - example:
     ```Plain text
        tuple<tuple<int, _6>, tuple<int, _3>, int> test(tuple<int, _6>(3, _6{}),
                                                        tuple<int, _3>(4, _3{}), 4);
        tuple<tuple<int, _6>, tuple<_3, _3>, int> test_right(tuple<int, _6>(2, _6{}),
                                                            tuple<_3, _3>(_3{}, _3{}), 2);
        print(shape_div(test, 10));          // ((1,2),(4,3),4)
        print(shape_div(test, test_right));  // ((1,_1),(1,_1),2)
    ```
   - code：
     ```c++
        template <class IntTupleA, class IntTupleB>
        CUTE_HOST_DEVICE constexpr
        auto
        shape_div(IntTupleA const& a, IntTupleB const& b)
        {
        if constexpr (is_tuple<IntTupleA>::value) {
            if constexpr (is_tuple<IntTupleB>::value) {  // tuple tuple
            static_assert(tuple_size<IntTupleA>::value == tuple_size<IntTupleB>::value, "Mismatched ranks");
            return transform(a, b, [](auto const& x, auto const& y) { return shape_div(x,y); });
            } else {                                     // tuple int
            auto const [result, rest] = fold(a, cute::make_tuple(cute::make_tuple(), b),
                [] (auto const& init, auto const& ai) {
                return cute::make_tuple(append(get<0>(init), shape_div(ai, get<1>(init))), shape_div(get<1>(init), ai));
                });
            return result;
            }
        } else
        if constexpr (is_tuple<IntTupleB>::value) {    // int tuple
            return shape_div(a, product(b));
        } else
        if constexpr (is_static<IntTupleA>::value && is_static<IntTupleB>::value) {
            static_assert(IntTupleA::value % IntTupleB::value == 0 || IntTupleB::value % IntTupleA::value == 0, "Static shape_div failure");
            return C<shape_div(IntTupleA::value, IntTupleB::value)>{};
        } else {                                       // int int
            //assert(a % b == 0 || b % a == 0);          // Waive dynamic assertion
            return a / b != 0 ? a / b : signum(a) * signum(b);  // Division with rounding away from zero
        }

        CUTE_GCC_UNREACHABLE;
        }
     ```

 - shape_min: 二元函数, 获取对应两个Shape中对应位置的最小值。
   - 当前还不支持多元素的Tuple处理，经支持单个元素Tuple等处理;
  
 - elem_scale: 二元函数, 根据右操作数维度数值，对左操作数对应维度进行数值扩展。
   - 如果左操作数是Tuple，则要求右操作数必须也是Tuple，且元素个数要求一致;
   - 如果左操作数是Integral，则会先对右操作数进行求积的处理，然后再对左操作数进行数值扩展。
   - example:
     ```Plain text
        tuple<tuple<int, _6>, tuple<int, _3>, int> test(tuple<int, _6>(3, _6{}),
                                                        tuple<int, _3>(4, _3{}), 4);
        tuple<tuple<int, _6>, tuple<_3, _3>, int> test_right(tuple<int, _6>(2, _6{}),
                                                                tuple<_3, _3>(_3{}, _3{}), 2);
        print(elem_scale(10, test));          // 8640
        print(elem_scale(test, test_right));  // ((6,_36),(12,_9),8)
    ```
   - code:
     ```c++
        template <class A, class B>
        CUTE_HOST_DEVICE constexpr
        auto
        elem_scale(A const& a, B const& b)
        {
        if constexpr (is_tuple<A>::value) {
            return transform(a, b, [](auto const& x, auto const& y) { return elem_scale(x,y); });
        } else {
            return a * product(b);
        }

        CUTE_GCC_UNREACHABLE;
        }
     ```

 - congruent:  二元函数, 判断左右操作数个数是否完全一致。
   - 包括嵌套Tuple也需要进行检查
   - example：
     ```Plain text
        tuple<tuple<int, _6>, tuple<int, _3>, int> test(tuple<int, _6>(3, _6{}),
                                                        tuple<int, _3>(4, _3{}), 4);
        tuple<tuple<int, _6>, tuple<_3, _3>, int> test_right(tuple<int, _6>(2, _6{}),
                                                            tuple<_3, _3>(_3{}, _3{}), 2);
        tuple<tuple<int, _6>, tuple<_3>, int> test_right_new(tuple<int, _6>(2, _6{}),
                                                            tuple<_3>(_3{}), 2);
        print(congruent(10, test));              // _0{}
        print(congruent(test, test_right));      // _1{}
        print(congruent(test, test_right_new));  // _0{}
    ```
   - code:
    ```c++
        template <class IntTupleA, class IntTupleB>
        CUTE_HOST_DEVICE constexpr
        auto
        congruent(IntTupleA const& a, IntTupleB const& b)
        {
            return bool_constant<is_same<decltype(repeat_like(shape(a),_0{})),
                                        decltype(repeat_like(shape(b),_0{}))>::value>{};
        }
        template <class A, class B>
        using is_congruent = decltype(congruent(declval<A>(), declval<B>()));
    ```

## 基本运算规则
 - coalesce:
   



获取打印信息：
cmake .. -DCUTLASS_DEBUG_TRACE_LEVEL=ON -DCUTLASS_ENABLE_TESTS=ON
make cutlass_test_unit_cute_core -j; ./test/unit/cute/core/cutlass_test_unit_cute_core

