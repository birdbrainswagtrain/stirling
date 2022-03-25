local i = 0;
local sum = 0;
while i < 10000 do
    i = i + 1
    sum = (sum + 25/5 - 2*2) % 123;
end
print(sum)
