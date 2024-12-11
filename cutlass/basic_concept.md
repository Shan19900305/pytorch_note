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

## Layout

  - 代码路径： include/cute/layout.hpp
  ```c++
     template <class Shape, class Stride = LayoutLeft::Apply<Shape> >
     struct Layout
         : private cute::tuple<Shape, Stride>   // EBO for static layouts
  ```

### 计算stride的方式
  - 主要分为两种方式，即列主序LayoutLeft和行主序LayoutRight。两者调用函数基本一直，只存在于最后构造Tuple的方式存在差异。列主序通过append进行顺序展开，列主序则通过prepend进行从后进行展开。
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