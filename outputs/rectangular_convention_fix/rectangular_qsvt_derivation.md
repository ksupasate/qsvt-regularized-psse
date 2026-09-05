# Rectangular QSVT Derivation

Let `A = U Sigma V^dagger` be the zero-padded rectangular contraction in the top-left block of the Julia unitary

`W_A = [[A, sqrt(I-AA^dagger)], [sqrt(I-A^dagger A), -A^dagger]]`.

The production sequence is

`P(phi_0), W_A, P(phi_1), W_A^dagger, P(phi_2), ...`

where `P(phi)` multiplies the encoded top subspace by `exp(i phi)` and its complement by `exp(-i phi)`.  For production-native PennyLane phases this sequence places an odd polynomial in the real top-left block.  For calibrated PyQSP symmetric phases from the plus-i scalar convention, the equivalent dense-Julia/PCPhase convention is obtained by adding `pi/2` to every phase.  The transformed odd polynomial then appears in the signed imaginary top-left block, with sign `(-1)^((d+1)/2)`.  Since the target degree is 255, the sign is positive and the extracted block is `imag(top-left)`.

The extracted rectangular block is therefore

`imag( <top| Q_phi(W_A) |top> ) = U P(Sigma) V^dagger`

for the target degree-255 mapped phases.  The padding modes have singular value zero and remain zero because the target polynomial is odd and `P(0)=0`.

Configuration: `db67f79cce4b0a67c78530c0a2a185b729f9d7a2ea6baf40ab50325266a13189`.
