from matplotlib import pyplot as plt

def circle(R):
    neighbourhood = []
    X = int(R) # R is the radius
    for x in range(-X,X+1):
        Y = int((R*R-x*x)**0.5) # bound for y given x
        for y in range(-Y,Y+1):
            neighbourhood.append((x, y))

    return neighbourhood

plt.plot(circle(10))
plt.show()