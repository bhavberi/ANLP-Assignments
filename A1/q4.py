import matplotlib.pyplot as plt

"""
LM3:
- Train: 8.375014625931152
- Val: 9.953790048247354
- Test: 9.79198909045301
LM2:
- Train: 104.4500672383349
- Val: 207.9000903323066
- Test: 201.9537698778273
LM1:
- Train: 143.02995201102905
- Val: 211.7381600513458
- Test: 211.42966722100843
"""

# Data
x = ['Train', 'Val', 'Test']
y1 = [143.02995201102905, 211.7381600513458, 211.42966722100843]
y2 = [104.4500672383349, 207.9000903323066, 201.9537698778273]
y3 = [8.375014625931152, 9.953790048247354, 9.79198909045301]

# Create a bar graph
# plt.bar(x, y1, color='b', width=0.25, label='LM1')
# plt.bar(x, y2, color='g', width=0.25, label='LM2')
# plt.bar(x, y3, color='r', width=0.25, label='LM3')
# Create side by side bar graph
barWidth = 0.25
r1 = range(len(x))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, y1, color='b', width=barWidth, edgecolor='grey', label='LM1 - NNLM')
plt.bar(r2, y2, color='g', width=barWidth, edgecolor='grey', label='LM2 - LSTM')
plt.bar(r3, y3, color='r', width=barWidth, edgecolor='grey', label='LM3 - Transformer')

# Add labels
plt.xlabel('Data')
plt.ylabel('Perplexity')
plt.title('Perplexity for different LMs')
plt.legend()

# Show the plot
# plt.savefig('perplexity.png')
plt.show()
