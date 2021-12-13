function pr = psnr(A, B)
val = 255
mse = mean((A - B) .* (A - B), 'all')

pr = 10 * log10(val ^ 2 / mse)
end