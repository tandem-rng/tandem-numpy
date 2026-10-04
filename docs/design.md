# Design

## Bounded integers

`below` draws on `[0, n)` by Lemire's multiply-and-reject over `uint32` or `uint64` draws
(`dtype`, default `uint64`). With `size` or `out` it draws element `i` from stream draw `i`
and retries a rejected draw on a fallback generator keyed by the draw's global index, so it
uses exactly one draw per element, the position advances by that, and a fill cut at any element
boundary equals the whole fill. Without them it is one scalar draw, which takes as many
draws as the rejection loop needs. `n = 0` returns 0.

## Normals

`normal` with `float64` is the 1024-layer ziggurat of Appendix A. Element `i` comes from 64-bit
draw `i`, so a fill equals the scalar draws and a fill cut at any element equals the whole fill.
A draw outside the inner rectangles, about 0.4 % of them, continues on a fallback stream keyed
by its global draw index. An empty fill aligns the position to 64.

`normal` with `float32` is Box-Muller on pairs of `float32` uniforms `a`, `b`: with
`r = sqrt(-2 ln(1 - a))` the pair is `(r cos 2 pi b, r sin 2 pi b)`. Pair `j` gives elements
`2j` (cos half) and `2j + 1` (sin half) from uniforms `2j` and `2j + 1`. An odd `size` keeps
the cos half of its last pair and still consumes both uniforms. A scalar draw is the cos half
and consumes two uniforms, so it equals element 0 of a fill.

The values of both are the bits of tandem-c's normal fills on every compiler.

## Exponentials

`exponential` draws `-ln(1 - u)` from one uniform `u` of the dtype per element, `float64` from
`float64` uniforms in double and `float32` from `float32` uniforms in single, with the
polynomial logarithm of the normals and no libm call. Element `i` comes from uniform `i`, so a
fill equals the scalar draws, a fill cut at any element boundary equals the whole fill, and
`n = 0` leaves the position unchanged. The bits are the same in every Tandem port and on every
compiler.
