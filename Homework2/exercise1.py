import matplotlib.pyplot as plt
import numpy as np

def statement(x1, x2, x3):
    return (x1 and x2 and (not x3)) or ((not x2) and x3)

def equation(x1, x2, x3):
    x = np.array([[x1], [x2], [x3]])
    return sign(np.dot(U, sign(np.dot(W, x) + b)) + c)

def translate(x):
    if x == True:
        return 1
    else:
        return -1

def translateBack(x):
    if x == 1:
        return True
    else:
        return False

def sign(v):
    result = np.array([])
    for i in v:
        if i > 0:
            result = np.append(result, 1)
        elif i == 0:
            result = np.append(result, 0)
        else:
            result = np.append(result, -1)
    return result

# data computed with statement logic
dataS = []
# data computed with analytic equation
dataE = []

values = [True, False]

W = np.array([[1, 1, -1], [0, -1, 1]])
b = np.array([[-2.5], [-1.5]])
U = np.array([1, 1])
c = np.array([0.5])

for x1 in values:
    for x2 in values:
        for x3 in values:
            resultS = statement(x1, x2, x3)
            resultE = equation(translate(x1), translate(x2), translate(x3))
            dataS.append([translate(x1), translate(x2), translate(x3), translate(resultS)])
            dataE.append([x1, x2, x3, translateBack(resultE[0])])

dataS = np.array(dataS)

figS, axS = plt.subplots(figsize=(8, 4))
axS.xaxis.set_visible(False)
axS.yaxis.set_visible(False)
axS.set_frame_on(False)

tableS = axS.table(cellText=dataS, colLabels=['x1', 'x2', 'x3', 'y=(x1∧x2∧¬x3)∨(¬x2∧x3)'], cellLoc='center', loc='center')
tableS.auto_set_font_size(False)
tableS.set_fontsize(12)
tableS.auto_set_column_width([0, 1, 2, 3])

for i, key in enumerate(tableS._cells.keys()):
    cell = tableS._cells[key]
    cell.set_height(0.1)


dataE = np.array(dataE)

figE, axE = plt.subplots(figsize=(8, 4))
axE.xaxis.set_visible(False)
axE.yaxis.set_visible(False)
axE.set_frame_on(False)

tableE = axE.table(cellText=dataE, colLabels=['x1', 'x2', 'x3', 'y=f(x)'], cellLoc='center', loc='center')
tableE.auto_set_font_size(False)
tableE.set_fontsize(12)
tableE.auto_set_column_width([0, 1, 2, 3])

for i, key in enumerate(tableE._cells.keys()):
    cell = tableE._cells[key]
    cell.set_height(0.1)

plt.show()

