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

### 辅助函数
 - for_each， transform, find， find_if， any_of， all_of， none_of， filter_tuple： 遍历所有元素，执行f函数。
 - for_each_leaf， transform_leaf：  遍历所有元素（支持嵌套结构的处理），执行f函数。
 - FoldAdaptor实现： 基于重载运算符，对序列TupleA进行展开逐个运算。
   - code:
     ```c++
      template <class Fn, class Val>
      struct FoldAdaptor {
        template <class X>
        CUTE_HOST_DEVICE constexpr auto operator|(X&& x) {
          auto r = fn_(val_, static_cast<X&&>(x));
          return FoldAdaptor<Fn, decltype(r)>{fn_, r};
        }
        Fn fn_;
        Val val_;
      };

      template <class T, class V, class F, int... Is>
      CUTE_HOST_DEVICE constexpr
      auto
      fold(T&& t, V const& v, F&& f, seq<Is...>)
      {
        return (FoldAdaptor<F,V>{f,v} | ... | get<Is>(static_cast<T&&>(t))).val_;
      }
     ```

## Tuple基本接口和运算
 - 获取Tuple相关属性: 通过index进行索引获取对应属性,支持多层index索引。
   - 形式： rank<0, 1>(tuple)
   - 具体： rank, shape, depth, size, front, back, insert, remove, replace, replace_front, replace_backtuple_repeat, repeat, repeat_like, group, append, prepend

 - 基础运算： 通过Tuple展开, 
   - 形式： min(tuple1, tuple2)
   - 两元输入场景： min, max, gcd, inner_product, round_up, shape_min, ceil_div, shape_div, round_up, congruent, weakly_congruent, elem_scale, compatible, evenly_divides, filter_zeros, iscan, escan
   - 形式： product(tuple1)
   - 单输入场景： product, sum, product_each, product_like

 - 基本元素操作：
   - 形式： select<start, end>(Tuple)
   - 具体： select, take, zip, make_int_tuple, fill_int_tuple_from, make_int_tuple_from, wrap/unwrap, flatten, unflatten

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

 - shape: 通过模板参数获取对于维度的元素信息,如果对应维度为嵌套的子容器,则返回子容器的shape。
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

 - size: 根据索引index获取对应嵌套结构的元素个数,内部调用product完成处理。
   - size example：
     ```Plain text
       tuple<tuple<int, _6>, tuple<int, _3>, int> test(tuple<int, _6>(3, _6{}),
                                                       tuple<int, _3>(4, _3{}), 4);
       size(test);       // 864
       size<0>(test);    // 18
       size<0, 1>(test); // _6{}
     ```

 - front： 获取Tuple中的第一个非Tuple元素。
 - back： 获取Tuple最后一个非Tuple元素。
 - insert / remove / replace / replace_front / replace_back： 对Tuple的第N个位置元素的插入，删除和替换。
 - tuple_repeat / repeat / repeat_like： 基于X构造一个新的Tuple。
 - group： 聚合Tuple中的部分元素，作为Tuple子元素，即为嵌套结构。
 - append / prepend： 在Tuple中添加元素。

 - iscan： 基于传入的函数f和v，对Tuple中的每个元素进行迭代。
   - 第I个元算置v_next = f(v, get<I>(t))，并使用v_next替换当前元素。
   - 将v_next作为新的v对第I+1个元素进行处理。
   - code:
      ```c++
        template <class T, class V, class F, int I, int... Is>
        CUTE_HOST_DEVICE constexpr
        auto
        iscan(T const& t, V const& v, F&& f, seq<I,Is...>)
        {
          // Apply the function to v and the element at I
          auto v_next = f(v, get<I>(t));
          // Replace I with v_next
          auto t_next = replace<I>(t, v_next);

        #if 0
          std::cout << "ISCAN i" << I << std::endl;
          std::cout << "  t      " << t << std::endl;
          std::cout << "  i      " << v << std::endl;
          std::cout << "  f(i,t) " << v_next << std::endl;
          std::cout << "  t_n    " << t_next << std::endl;
        #endif

          if constexpr (sizeof...(Is) == 0) {
            return t_next;
          } else {
            return iscan(t_next, v_next, f, seq<Is...>{});
          }

          CUTE_GCC_UNREACHABLE;
        }
      ```

 - escan 基于传入的函数f和v，对Tuple中的每个元素进行迭代。
   - 第I个元算置v_next = f(v, get<I>(t))，并使用v替换当前元素。
   - 将v_next作为新的v对第I+1个元素进行处理。
   - code:
      ```c++
        template <class T, class V, class F, int I, int... Is>
        CUTE_HOST_DEVICE constexpr
        auto
        escan(T const& t, V const& v, F&& f, seq<I,Is...>)
        {
          if constexpr (sizeof...(Is) == 0) {
            // Replace I with v
            return replace<I>(t, v);
          } else {
            // Apply the function to v and the element at I
            auto v_next = f(v, get<I>(t));
            // Replace I with v
            auto t_next = replace<I>(t, v);

        #if 0
            std::cout << "ESCAN i" << I << std::endl;
            std::cout << "  t      " << t << std::endl;
            std::cout << "  i      " << v << std::endl;
            std::cout << "  f(i,t) " << v_next << std::endl;
            std::cout << "  t_n    " << t_next << std::endl;
        #endif

            // Recurse
            return escan(t_next, v_next, f, seq<Is...>{});
          }

          CUTE_GCC_UNREACHABLE;
        }
      ```

 - zip： 将多个Tuple中的元素进行一对一的打包,生成一个新的嵌套Tuple。
   - example:
    ```Plain text
      auto zip_test_shape1 = make_shape(Int<128>{}, Int<64>{}, Int<62>{});
      auto zip_test_shape2 = make_shape(Int<127>{}, Int<63>{}, Int<61>{});
      auto zip_test = zip(zip_test_shape1, zip_test_shape2);
      result --> zip_test: ((_128,_127),(_64,_63),(_62,_61))
    ```
   - code:
    ```c++
      // include/cute/algorithm/tuple_algorithms.hpp
      template <class T, int... Is, int... Js>
      CUTE_HOST_DEVICE constexpr
      auto
    zip(T const& t, seq<Is...>, seq<Js...>)
      {
        static_assert(conjunction<bool_constant<tuple_size<tuple_element_t<0,T>>::value ==  tuple_size<tuple_element_t<Is,T>>::value>...>::value, "Mismatched Ranks");
        // get<Is>(t)... 将Tuple拆分成单对单的tuple
        // zip_<Js>(get<Is>(t)...) 对应取每一个Tuple中的第Js个元素,并打包成一个tuple
        return cute::make_tuple(zip_<Js>(get<Is>(t)...)...);
      }
    ```

 - zip2_by： 输入TupleA和TupleB，根据TupleB的元素个数，对TupleA进行解包处理。
   - 类似输入TupleA： ((A,a),((B,b),(C,c)),d) 基于TupleB的元素个数，获得((A,(B,C)),(a,(b,c),d))
   - code:
      ```c++
        namespace detail {

        template <class T, class TG, int... Is, int... Js>
        CUTE_HOST_DEVICE constexpr
        auto
        zip2_by(T const& t, TG const& guide, seq<Is...>, seq<Js...>)
        {
          // zip2_by produces the modes like ((A,a),(B,b),...)
          auto split = cute::make_tuple(zip2_by(get<Is>(t), get<Is>(guide))...);

          // Rearrange and append missing modes from t to make ((A,B,...),(a,b,...,x,y))
          return cute::make_tuple(cute::make_tuple(get<0>(get<Is>(split))...),
                                  cute::make_tuple(get<1>(get<Is>(split))..., get<Js>(t)...));
        }

        } // end namespace detail

        template <class T, class TG>
        CUTE_HOST_DEVICE constexpr
        auto
        zip2_by(T const& t, TG const& guide)
        {
          if constexpr (is_tuple<TG>::value) {
            constexpr int TR = tuple_size<T>::value;
            constexpr int GR = tuple_size<TG>::value;
            static_assert(TR >= GR, "Mismatched ranks");
            return detail::zip2_by(t, guide,
                                  make_range< 0, GR>{},
                                  make_range<GR, TR>{});
          } else {
            static_assert(tuple_size<T>::value == 2, "Mismatched ranks");
            return t;
          }

          CUTE_GCC_UNREACHABLE;
        }
      ```
    - example：
       ```Plain text
         zip2_by(tuple<tuple<_2, _2>, _3>{}, tuple<_1>{})  // ((_2), (_2, _3))
       ```

 - reverse： 翻转Tuple。 

 - max / min / gcd: 遍历所有维度元素,然后取最大值 / 最小值 / 最大公约数。其中嵌套结构会被展开铺平。
   - 不支持获取特定嵌套结构或维度的最大值 / 最小值 / 最大公约数,需要通过shape获取对应维度后进行最大值获取。
   - 支持同时传入多个shape,然后取最大值 / 最小值 / 最大公约数。
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
   - 可以简单理解为将tuple中的子tuple序列规约为单个数值或者保持不变。其中会严格对其每个tuple的元素个数,如果元素个数不一致,则编译器会报错。
   - product_like example1:
     ```Plain text
        tuple<tuple<int, _2>, int, int>>tuple_3h_m(tuple<int,_2>(1,_2{}), 8, 2);
        tuple<int, int, _2> tuple_3d_m(8,4,_2{});
        auto result = product_like(tuple_3h_m, tuple_3d_m); // tuple<int, int, int> result{2, 8, 2};
     ```

 - sum: 进行求和的处理,存在嵌套结构时,则会进行嵌套求和。
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

 - inner_product: 二元函数,计算两个Tuple对应元素乘积后的累加和。
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
   - 如果左右操作数都为Tuple,则要求左操作数的元素个数要大于等于右操作数。首先进行右操作数维度补齐（尾端补1）,然后展开左右操作数进行Integral的除法操作;
   - 如果左操作数是Tuple,右操作数是Integral,首元素与右操作数进行Integral的除法操作;
     - 主要逻辑运算集中在fold的处理,具体如下：
       - 第一个元素和Integral向上取整除法,作为第一个输出; Integral和第一个元素向上取整除法，作为余数;
       - 对应运算的余数和第二个元素进行运算,重复第一个元素的运算步骤,直到对应Tuple或嵌套内层Tuple所有元素都处理完毕。
   - 如果左操作数是Integral,右操作数是Tuple,则先对右操作数进行product操作,然后进行Integral的除法操作;
   - 如果左右操作数都是Integral,则直接进行向上取整的除法操作。
   - example：
     ```Plain text
        tuple<tuple<int, _6>, tuple<int, _3>, int> test(tuple<int, _6>(3, _6{}),
                                                        tuple<int, _3>(4, _3{}), 4);
        tuple<tuple<int, _6>, tuple<_3>, int> test_right(tuple<int, _6>(2, _6{}),
                                                            tuple<_3>(_3{}), 2);
        print(ceil_div(test, 3));           // ((1, 6), (4, _3), 4)
        print(ceil_div(test, test_right));  // ((2, _1), (2, _3), 2)
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
   - 仅支持左右操作数都为Tuple或Intiger类型。如果是Tuple且存在嵌套结构,需要满足元素个数一致。如果不存在嵌套结构且元素个数不一致,则进行尾部补1的处理。
   - 元素对齐场景,进行对为的round_up操作。
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

 - shape_div: 二元函数, 对应位置元素进行左操作数对右操作数进行取整处理。代码处理与ceil_div基本一致。
   - 如果左右操作数都为Tuple,严格要求左右操作数元素个数一致,逐元素进行div运算;
   - 如果左操作数是Tuple,右操作数是Integral,首元素与右操作数进行Integral的除法操作;
     - 主要逻辑运算集中在fold的处理,具体如下：
       - 第一个元素除右操作数,取对应运算结果作为结果的第一个元素;
       - 右操作数处以第一个元素,取对应运算结果intermedia;
       - 对应运算结果intermedia和第二个元素进行运算,重复第一个元素的运算步骤,直到对应Tuple或嵌套内层Tuple所有元素都处理完毕。
   - 如果左操作数是Integral,右操作数是Tuple,则先对右操作数进行product操作,然后进行Integral的除法操作;
   - 如果左右操作数都是Integral,则直接进行Integral的除法操作。
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
   - 当前还不支持多元素的Tuple处理,经支持单个元素Tuple等处理;
  
 - elem_scale: 二元函数, 根据右操作数维度数值,对左操作数对应维度进行数值扩展。
   - 如果左操作数是Tuple,则要求右操作数必须也是Tuple,且元素个数要求一致;
   - 如果左操作数是Integral,则会先对右操作数进行求积的处理,然后再对左操作数进行数值扩展。
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

 - congruent: 二元函数, 判断左右操作数元素个数是否完全一致。如果存在嵌套Tuple, 会进行嵌套Tuple展开后逐一对比。
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

 - weakly_congruent： 二元函数, 判断左右操作数元素个数是否完全一致。如果存在嵌套Tuple, 会进行嵌套Tuple展开后逐一对比。
   - 当TupleA为（或对应rank的元素为）Interger类型时，则忽略对于TupleB（或对应rank元素）的检查和判断。
   - code：
     ```c++
      template <class IntTupleA, class IntTupleB>
      CUTE_HOST_DEVICE constexpr
      auto
      weakly_congruent(IntTupleA const& a, IntTupleB const& b)
      {
        if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
          if constexpr (tuple_size<IntTupleA>::value != tuple_size<IntTupleB>::value) {
            return false_type{};
          } else {
            return transform_apply(a, b, [](auto const& x, auto const& y) { return weakly_congruent(x,y); },
                                        [](auto const&... z) { return (true_type{} && ... && z); });
          }
        } else if constexpr (is_integral<IntTupleA>::value) {
          return true_type{};
        } else if constexpr (is_integral<IntTupleB>::value) {
          return false_type{};
        } else {
          return weakly_congruent(shape(a), shape(b));
        }

        CUTE_GCC_UNREACHABLE;
      }
     ```

 - take: 基于Begin和End截取Tuple中的部分数据,其中左闭右开的区间。
   - code:
     ```c++
       template <int B, int E, class T>
       CUTE_HOST_DEVICE constexpr
       auto
       take(T const& t)
       {
         if constexpr (E == -1) {
           if constexpr (is_tuple<T>::value) {
             return take<B,tuple_size<T>::value>(t);
           } else {
             return take<B,1>(t);
           }
         } else
         if constexpr (B <= E) {
           return detail::apply(t, [](auto const&... a) { return cute::make_tuple(a...); }, make_range<B,E>{});
         } else {
           static_assert(B <= E);
         }

         CUTE_GCC_UNREACHABLE;
       }
     ```

 - select: 基于给定的index序列,获取对应位置的Tuple元算。
   - code:
     ```c++
       template <int... I, class T>
       CUTE_HOST_DEVICE constexpr
       auto
       select(T const& t)
       {
         return cute::make_tuple(get<I>(t)...);
       }
     ```

 - compatible: 判断两个Tuple是否满足如下两个特点：
   - 如果都为Tuple时，要求两者rank大小相同，对比每一个rank下的维度数值一致;
   - 如果TupleA为Integral时，要求TupleB的product和左操作数相同。
   - 满足上述条件后，则TupleA和TupleB拥有等价的coordinate，也就是A的coordinate可以通过作用于B。
   - code:
     ```c++
        template <class IntTupleA, class IntTupleB>
        CUTE_HOST_DEVICE constexpr
        auto
        compatible(IntTupleA const& a, IntTupleB const& b)
        {
          if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
            if constexpr (tuple_size<IntTupleA>::value != tuple_size<IntTupleB>::value) {
              return false_type{};
            } else {
              return transform_apply(a, b, [](auto const& x, auto const& y) { return compatible(x,y); },
                                          [](auto const&... z) { return (true_type{} && ... && z); });
            }
          } else if constexpr (is_integral<IntTupleA>::value) {
            return a == size(b);
          } else if constexpr (is_integral<IntTupleB>::value) {
            return false_type{};
          } else {
            return compatible(shape(a), shape(b));
          }

          CUTE_GCC_UNREACHABLE;
        }
     ```
   - example:
     ```plain text
     
     ```

 - evenly_divides： 二元运算，表示TupleA可以被TupleB均匀地分割。
   - if result is true_type, then size(a) == logical_divide(make_layout(shape(a)),b) will always compile and result in true_type.
   - code:
     ```c++
      template <class Shape, class Tiler>
      CUTE_HOST_DEVICE constexpr
      auto
      evenly_divides(Shape const& a, Tiler const& b)
      {
        if constexpr (is_tuple<Tiler>::value) {
          if constexpr (rank_v<Tiler> > rank_v<Shape>) {
            return false_type{};
          } else {
            return transform_apply(b, a, [](auto const& x, auto const& y) { return evenly_divides(y,x); },
                                        [](auto const&... z) { return (true_type{} && ... && z); });
          }
        } else {
          return size(a) == size(b) * size(ceil_div(shape(a), b));
        }

        CUTE_GCC_UNREACHABLE;
      }
     ```

 - filter_zeros： 二元运算，将Tuple中的0元素替换为1元素。
   - code：
    ```c++
      template <class IntTupleA, class IntTupleB>
      CUTE_HOST_DEVICE constexpr
      auto
      filter_zeros(IntTupleA const& a, IntTupleB const& b)
      {
        if constexpr (is_tuple<IntTupleA>::value) {
          return transform(a, b, [](auto const& x, auto const& y) { return filter_zeros(x,y); });
        } else if constexpr (is_constant<0, IntTupleA>::value) {
          return repeat_like(b, Int<1>{});
        } else {
          return b;
        }

        CUTE_GCC_UNREACHABLE;
      }

      template <class Tuple>
      CUTE_HOST_DEVICE constexpr
      auto
      filter_zeros(Tuple const& t)
      {
        return filter_zeros(t, t);
      }
    ```

 - make_int_tuple： 基于传入容器和默认值生成一个指定元素个数的IntTuple。
 - fill_int_tuple_from： 通过TupleB中数值替换TupleA中非常量元素。
 - make_int_tuple_from： 基于传入序列，生成一个Tuple。
 - wrap/unwrap: 将整数打包成rank=1的tuple,或者将rank=1的tuple解包成整数。
 - flatten： 将所有的嵌套Tuple解包合并到一个单个Tuple。
 - unflatten： 根据目标Tuple的结构,对当前Tuple进行相同方式的打包。

 - Comparison operators
   - 采用字节序的大小对比方式。其中行模式为：lexicographical comparison，函数名称为lex_less。列模式为：colexicographical comparison，函数名称为colex_less。普遍的elementwise对比，函数名称为elem_less。
     - 要求对应rank对应元素类型一致，要么都是Tuple，要么都是Integral。
     - lex_less： 逐元素对比TupleA和TupleB，其主要给行模式场景使用。
       - 相对TupleA，如果TupleB中对应元素耗尽，或者对应嵌套Tuple耗尽，则返回false。反之返回ture。
       - 如果TupleA和TupleB中元素一致，则基于元素值进行逐元素对比。如果最后一个元素小于TupleB中元素，且其他元素都小于或等于TupleB中元素，则返回true。
       - example：
         ```plain text
           lex_less(tuple<tuple<_2, _2>, _2>{}, tuple<tuple<_2,_2>, _3>{}) // true
           lex_less(tuple<tuple<_2, _2>, _2>{}, tuple<tuple<_2,_2>, _2>{}) // false
           lex_less(tuple<tuple<_2, _2>, _2>{}, tuple<tuple<_2,_3>, _2>{}) // true
           lex_less(tuple<tuple<_2, _2>, _2>{}, tuple<tuple<_2>, _3>{})    // false
           lex_less(tuple<tuple<_2>, _2>{}, tuple<tuple<_2,_2>, _3>{})     // true
         ```
     - lex_leq / lex_gtr / lex_geq: lex_leq即小于等于，gtr即大于，geq即大于等于。均是通过lex_less变换得到。
     - colex_less： 相对处理逻辑和lex_less一致，其主要给列模式场景使用。
       - 相对lex_less，其从最后一个维度开始向前对比。
       - example：
         ```plain text
           colex_less(tuple<tuple<_2, _2>, _2>{}, tuple<tuple<_2,_2>, _3>{}) // true
           colex_less(tuple<tuple<_2, _2>, _2>{}, tuple<tuple<_2,_2>, _2>{}) // false
           colex_less(tuple<tuple<_2, _2>, _2>{}, tuple<tuple<_2,_3>, _2>{}) // true
           colex_less(tuple<tuple<_2, _2>, _2>{}, tuple<tuple<_2>, _3>{})    // true
           colex_less(tuple<tuple<_2>, _2>{}, tuple<tuple<_2,_2>, _3>{})     // true
         ```
    - colex_leq / colex_gtr / colex_geq： 均是通过colex_less变换得到。
    - elem_less： 逐元素对比TupleA和TupleB，其处理逻辑为：
      - Rank不一致的情况下，如果TupleA中对应元素耗尽，返回True，反之返回ture。
      - 对应要求每个元素都满足小于的逻辑关系。
       - example：
         ```plain text
           elem_less(tuple<tuple<_1, _1>, _2>{}, tuple<tuple<_2,_2>, _3>{}); // true
           elem_less(tuple<tuple<_2, _2>, _2>{}, tuple<tuple<_2,_2>, _2>{}); // false
           elem_less(tuple<tuple<_2, _2>, _2>{}, tuple<tuple<_2,_3>, _2>{}); // false
           elem_less(tuple<tuple<_1, _1>, _2>{}, tuple<tuple<_2>, _3>{});    // false
           elem_less(tuple<tuple<_1>, _1>{}, tuple<tuple<_2,_2>, _3>{});     // true
         ```

获取打印信息：
cmake .. -DCUTLASS_DEBUG_TRACE_LEVEL=ON -DCUTLASS_ENABLE_TESTS=ON -DCUTLASS_NVCC_ARCHS="75"
make cutlass_test_unit_cute_core -j; ./test/unit/cute/core/cutlass_test_unit_cute_core

