# Fast Furier Transform


The furier transofrm equations:


$$H(f) = \int_{-\infty}^{\infty} h(t) e^{-2\pi i f t} dt$$

can also be written as

$$h(f) = \int_{-\infty}^{\infty} H(f) e^{-2\pi i f t} dt$$


- t is the time domain, given in seconds
- f is the frequency domain, given in hertz (cycles per second)

(sometimes also x and k where x is in meters and k in cycles / meter)

## Properties

- The furier transform is linear, meaning that the transform of a sum is the sum of the transforms. $h(t) + g(t) = H(f) + G(f)$

- Symmetries: 
    - $h(t)$ real then  $H(-f) = [H(f)]^*$
    - $h(t)$ imaginary then $H(-f) = -[H(f)]^*$
    - $h(t)$ odd and $h(t) = -h(-t)$ then $H(-f) = -H(f)$ where $H$ is odd
    - $h(t)$ even and $h(t) = h(-t)$ then $H$ is even

Theese can speed up the computation of the furier transform.
Additionally there are other properties such as time and frequency scaling and time and frequency shifting and the convolution theorem.

## Discrete furier transform

The furier transform is defined for continuous functions, but we can also define it for discrete functions. The furier transform of a discrete function is called the discrete furier transform (DFT).

The DFT is defined as:

$$H(k) = \sum_{n=0}^{N-1} h(n) e^{-2\pi i \frac{kn}{N}}$$

where $k$ is the frequency index and $n$ is the time index. $N$ is the number of samples.

Nyquist frequency: $f_{Ny} = \frac{1}{2\Delta t}$

## Sampling Theorem

The sampling theorem states that a signal can be perfectly reconstructed from its Fourier transform if the following conditions are met:

- The signal is bandlimited, i.e. the signal has a finite bandwidth.
- The sampling frequency is at least twice the bandwidth of the signal.

## Fast Furier Transform

The fast furier transform (FFT) is an algorithm to compute the DFT. It is based on the fact that the DFT can be written as a product of two matrices. The first matrix is a diagonal matrix with the values $e^{-2\pi i \frac{kn}{N}}$ and the second matrix is a diagonal matrix with the values $h(n)$.

Given a vector $y$


- N^2 method
- NlogN recursive method
- NlogN 