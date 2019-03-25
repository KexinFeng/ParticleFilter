f = [.3054 .5243 .1713; .3222 .5395 .1383; 0 .2465 .7535 ];
f1 = f(:,1);
f2 = f(:, 2);
f3 = f(:, 3);

b = [.2374 .5836 .0627; .1178 .5787 .0311; 0 .2465 .7535];
b1 = b(:, 1);
b2 = b(:, 2);
b3 = b(:, 3);

colors = linspecer(3);
% color = {};
% for i = 1:size(colors, 1)
%     color = [color, {colors(i, :)}];
% end

x = 1:3;
figure
h1 = plot(x, f1,'-o', x, f2,'-o', x, f3, '-o');
set(h1, {'color'}, num2cell(colors, 2));
hold on
h2 = plot(x, b1,'--*', x, b2, '--*',x, b3, '--*');
set(h2, {'color'}, num2cell(colors, 2));

legend({'filter: low', 'filter: medium', 'filter: high','smooth: low', 'smooth: medium', 'smooth: high'},'location','Northwest');
title('Filtering and smoothed probability of x_t, E = [~e1, ~e2, e3]');
xlabel('t');
ylabel('probability of x_t');

set(gcf,'unit','centimeters','position',[4, 1, 23, 15]);
saveas(gcf, 'plotting.fig');
saveas(gcf, 'plotting.jpg');
saveas(gcf, 'plotting.eps', 'psc2');