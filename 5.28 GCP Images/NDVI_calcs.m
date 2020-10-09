imtool(T16)
redChannel = T16(:,:,1);
blueChannel = T16(:,:,3);
prompt = 'What is the pixel region?';
x = input(prompt);
a = x(1,1);
b = x(1,2);
c = x(1,3);
d = x(1,4);
avg_B = mean2(blueChannel(a:(a+c),b:(b+d)))
avg_R = mean2(redChannel(a:(a+c),b:(b+d)))
NDVI = (1.706*avg_B - 0.706*avg_R) / (0.294*avg_B + 0.706*avg_R)

