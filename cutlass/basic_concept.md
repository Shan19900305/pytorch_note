# Basic Concept
从0开始学习基本概念

## 基于Tuple构造的基本概念
基于Tuple类进行别名处理，构建了基本存储容器Shape,Stride,Step,Coord,Tile。
注： 此处的Coord和cutlass中的Coord类存在差别，待确认。
  ```c++
    template <class... Shapes>
    using Shape = cute::tuple<Shapes...>;

    template <class... Strides>
    using Stride = cute::tuple<Strides...>;

    template <class... Strides>
    using Step = cute::tuple<Strides...>;

    template <class... Coords>
    using Coord = cute::tuple<Coords...>;

    template <class... Layouts>
    using Tile = cute::tuple<Layouts...>;
   ```

## Stride
  - 代码路径： include/cute/stride.hpp
  - crd2idx函数，通过Coord坐标信息，计算对应Layout下真实的内存偏移量。
    - code：
      ```c++
        template <class Coord, class Shape, class Stride>
        CUTE_HOST_DEVICE constexpr
        auto
        crd2idx(Coord  const& coord,
                Shape  const& shape,
                Stride const& stride)
        {
          if constexpr (is_tuple<Coord>::value) {
            if constexpr (is_tuple<Shape>::value) {      // tuple tuple tuple
              static_assert(tuple_size<Coord>::value == tuple_size< Shape>::value, "Mismatched Ranks");
              static_assert(tuple_size<Coord>::value == tuple_size<Stride>::value, "Mismatched Ranks");
              // 展开计算每一个坐标点对应的offset，然后进行累加。
              return detail::crd2idx_ttt(coord, shape, stride, tuple_seq<Coord>{});
            } else {                                     // tuple "int" "int"
              static_assert(sizeof(Coord) == 0, "Invalid parameters");
            }
          } else {
            if constexpr (is_tuple<Shape>::value) {      // "int" tuple tuple
              static_assert(tuple_size<Shape>::value == tuple_size<Stride>::value, "Mismatched Ranks");
              // 基于coord对shape进行求整数段和余数段，余数段*对应维度stride；
              // 整数段和下一个shape进行余数段和整数段计算，直到对应shape遍历完。
              return detail::crd2idx_itt(coord, shape, stride, tuple_seq<Shape>{});
            } else {                                     // "int" "int" "int"
              return coord * stride;
            }
          }

          CUTE_GCC_UNREACHABLE;
        }

        template <class CInt, class STuple, class DTuple, int I0, int... Is>
        CUTE_HOST_DEVICE constexpr
        auto
        crd2idx_itt(CInt   const& coord,
                    STuple const& shape,
                    DTuple const& stride, seq<I0,Is...>)
        {
          if constexpr (sizeof...(Is) == 0) {  // Avoid recursion and mod on single/last iter
            return crd2idx(coord, get<I0>(shape), get<I0>(stride));
          } else if constexpr (is_constant<0, CInt>::value) {
            return crd2idx(_0{}, get<I0>(shape), get<I0>(stride))
                + (_0{} + ... + crd2idx(_0{}, get<Is>(shape), get<Is>(stride)));
          } else {                             // General case
            auto [div, mod] = divmod(coord, product(get<I0>(shape)));
            return crd2idx(mod, get<I0>(shape), get<I0>(stride))
                + crd2idx_itt(div, shape, stride, seq<Is...>{});
          }

          CUTE_GCC_UNREACHABLE;
        }
      ```

## Layout

  - 代码路径： include/cute/layout.hpp
  - 大多数场景按照默认列模式的方式进行构造Layout。
  ```c++
     template <class Shape, class Stride = LayoutLeft::Apply<Shape> >
     struct Layout
         : private cute::tuple<Shape, Stride>   // EBO for static layouts
  ```
  - 相关函数介绍（待补充相关函数细节）
    - 计算layout总的维度形状或对应单个维度的形状：shape
    - 计算layout总的维度步长或对应单个维度的步长：stride
    - 计算layout总的元素个数或对应单个维度元素个数： size
    - 计算layout总的维度数或对应单个维度的维度数： rank
    - 计算layout总的嵌套深度或对应单个维度的嵌套深度： depth
    ** 待补充coshape和cosize的区别，代码上是比较明确的，但是理解上两个是一个东西。不知道哪里理解错误了。**
    - 计算layout或者对应维度的物理内存空间大小（将stride计算进去）：coshape
      - code：
        ```c++
          template <int... Is, class Shape, class Stride>
          CUTE_HOST_DEVICE constexpr
          auto
          coshape(Layout<Shape,Stride> const& layout)
          {
            // Protect against negative strides
            auto abs_sub_layout = make_layout(shape<Is...>(layout),
                                              transform_leaf(stride<Is...>(layout), abs_fn{}));
            auto co_coord = as_arithmetic_tuple(abs_sub_layout(size(abs_sub_layout) - Int<1>{}));
            return co_coord + repeat_like(co_coord, Int<1>{});
          }
          // 主要通过operator()重载实现
          template <class Coord>
          CUTE_HOST_DEVICE constexpr
          auto
          operator()(Coord const& coord) const {
            if constexpr (has_underscore<Coord>::value) {
              return slice(coord, *this);
            } else {
              return crd2idx(coord, shape(), stride());
            }

            CUTE_GCC_UNREACHABLE;
          }

          // Convenience function for multi-dimensional coordinates
          template <class Coord0, class Coord1, class... Coords>
          CUTE_HOST_DEVICE constexpr
          auto
          operator()(Coord0 const& c0, Coord1 const& c1, Coords const&... cs) const {
            return operator()(make_coord(c0,c1,cs...));
          }
        ```
    - 计算layout或者对应维度的物理内存空间大小（将stride计算进去）：cosize
      - code：
        ```c++
          template <int... Is, class Shape, class Stride>
          CUTE_HOST_DEVICE constexpr
          auto
          cosize(Layout<Shape,Stride> const& layout)
          {
            return size(coshape<Is...>(layout));
          }
        ```
    - 构造Layout函数： make_layout
    - 按照给定序列顺序进行stride生产构造： make_ordered_layout(Shape const& shape, Order const& order)
      - example:
        ```Plain text
           make_ordered_layout(Shape<_2,_2,_2,_2>{}, Step<_0,_2,_3,_1>{})
           ->  (_2,_2,_2,_2):(_1,_4,_8,_2)
           make_ordered_layout(make_shape(2,3,4,5), make_step(Int<2>{}, 67, 42, Int<50>{}))
           -> (2,3,4,5):(_1,10,30,2)
           // shape index   shape value  new order   ref order      stride          produced stride
           //    0              2         Int<2>{}    Int<2>{}    {_1,_1, _1, _1}      _1
           //    1              3         Int<52>{}   Int<52>{}   {2,_1, _1, 5}        10
           //    2              4         Int<53>{}   Int<53>{}   {2, 3, _1, 5}        30
           //    3              5         Int<50>{}   Int<50>{}   {2, _1, _1, _1}       2
        ```
      - code：
        ```c++
          template <class Shape, class Order>
          CUTE_HOST_DEVICE constexpr
          auto
          make_ordered_layout(Shape const& shape, Order const& order)
          {
            return make_layout(shape, compact_order(shape, order));
          }
          // 调用compact_order，核心处理逻辑。
          // 1）先平铺shape和order，然后通过max_order获取最大静态order中的最大静态变量;
          // 2）基于最大常量生产一个新的max_seq，和order保持相同rank;
          // 3）根据max_seq和order生成新的order，将原有order中的非常量替换为对应位置的max_seq中的元素;
          // 4）compact_order计算获取最新的stride。
          template <class Shape, class Order>
          CUTE_HOST_DEVICE constexpr
          auto
          compact_order(Shape const& shape, Order const& order)
          {
            auto ref_shape = flatten_to_tuple(product_like(shape, order));

            auto flat_order = flatten_to_tuple(order);
            // Find the largest static element of order
            auto max_order = cute::fold(flat_order, Int<0>{}, [](auto v, auto order) {
              // 如果是C<a>和C<b>对比，则返回C<a < b>, 此时如果a<b，返回order，反之则返回v;
              // 如果是v或order不为C<x>对象，则返回v。
              if constexpr (is_constant<true, decltype(v < order)>::value) {
                return order;
              } else {
                return v;
              }

              CUTE_GCC_UNREACHABLE;
            });
            // Replace any dynamic elements within order with large-static elements
            auto max_seq = make_range<max_order+1, max_order+1+rank(flat_order)>{};
            // order为常量，则返回order;
            // order为变量，则返回seq_v;
            auto ref_order = cute::transform(max_seq, flat_order, [](auto seq_v, auto order) {
              if constexpr (is_static<decltype(order)>::value) {
                return order;
              } else {
                return seq_v;
              }

              CUTE_GCC_UNREACHABLE;
            });

            auto new_order = unflatten(ref_order, order);

            return detail::compact_order(shape, new_order, ref_shape, ref_order);
          }

        ```
    - 生成一个相同的Layout： make_layout_like
    - make_fragment_like

### 计算stride的方式
  - 主要分为两种方式，即列主序LayoutLeft和行主序LayoutRight。两者调用函数基本一直，只存在于最后构造Tuple的方式存在差异。列主序通过append进行顺序展开，列主序则通过prepend进行从后进行展开。
  - 代码路径： include/cute/stride.hpp
    ```c++
      template <class Major, class Shape, class Current>
      CUTE_HOST_DEVICE constexpr
      auto
      compact(Shape   const& shape,
              Current const& current)
      {
        if constexpr (is_tuple<Shape>::value) { // Shape::tuple Current::int
          using Lambda = CompactLambda<Major>;                  // Append or Prepend
          using Seq    = typename Lambda::template seq<Shape>;  // Seq or RSeq
          return cute::detail::fold(shape, cute::make_tuple(cute::make_tuple(), current), Lambda{}, Seq{});
        } else {                                // Shape::int Current::int
          if constexpr (is_constant<1, Shape>::value) {
            return cute::make_tuple(Int<0>{}, current); // If current is dynamic, this could save a reg
          } else {
            return cute::make_tuple(current, current * shape);
          }
        }

        CUTE_GCC_UNREACHABLE;
      }
      // 其中CompactLambda的处理就是通过乘法逐步向后进行依次处理。
      template <>
      struct CompactLambda<LayoutLeft>
      {
        template <class Init, class Shape>
        CUTE_HOST_DEVICE constexpr auto
        operator()(Init const& init, Shape const& si) {
          auto result = detail::compact<LayoutLeft>(si, get<1>(init));
          return cute::make_tuple(append(get<0>(init), get<0>(result)), get<1>(result));  // Append
        }

        template <class Shape>
        using seq = tuple_seq<Shape>;                                                     // Seq
      };
    ```


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


## Coord
Coord是表示一个坐标的模板类，
  - 代码路经： include/cutlass/coord.h
  - 其主要存在三个模板参数：kRank，Index，LongIndex; 
  - 其内存通过数组进行坐标点存储，kRank表示坐标的维度，Index表示坐标的元素类型，LongIndex表示坐标点的偏移量。
  - Coord提供了一系列的数学计算，方便进行坐标值的修改。


# 辅助工具及方法
## 单测试用例执行方法