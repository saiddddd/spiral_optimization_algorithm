import numpy as np

def objective_function(x):
    return rosenbrock_error(x, len(x))

def rosenbrock_error(x, dim):
    # Fungsi Rosenbrock error
    z = 0.0
    for i in range(dim-1):
        a = 100 * ((x[i+1] - x[i]**2)**2)
        b = (1 - x[i])**2
        z += a + b
    err = (z - 0.0)**2
    return err

def find_center(points):
    # Temukan pusat berdasarkan titik-titik dengan error terkecil
    best_err = objective_function(points[0])
    idx = 0
    for i, point in enumerate(points):
        err = objective_function(point)
        if err < best_err:
            idx = i
            best_err = err
    return np.copy(points[idx])

def move_point(x, R, RminusI, center):
    # Pindahkan titik x ke posisi baru, berputar mengelilingi pusat
    offset = np.matmul(RminusI, center)
    new_x = np.matmul(R, x) - offset
    return new_x

def main():
    print("\nBegin spiral dynamics optimization demo ")
    print("Goal is to minimize objective function")

    # Parameter algoritma
    theta = np.pi/3  # radians. small = curved, large = squared
    r = 0.98  # closer to 1 = tight spiral, closer 0 = loose 
    m = 50    # number of points / possible solutions
    n = 3     # problem dimension
    max_iter = 1000

    print("\nSetting theta = %0.4f " % theta)
    print("Setting r = %0.2f " % r)
    print("Setting number of points m = %d " % m)
    print("Setting max_iter = %d " % max_iter)

    # 1. set up the Rotation matrix for n=3
    print("\nSetting up hard-coded spiral Rotation matrix R ")

    ct = np.cos(theta)
    st = np.sin(theta)
    R12 = np.array([[ct,  -st,    0],
                  [st,   ct,    0],
                  [0,     0,    1]])

    R13 = np.array([[ct,   0,   -st],
                  [0,    1,     0],
                  [st,   0,    ct]])

    R23 = np.array([[1,    0,     0],
                  [0,    ct,  -st],
                  [0,    st,   ct]])

    R = np.matmul(R23, np.matmul(R13, R12))  # compose
    R = r * R  # scale / shrink

    I3 = np.array([[1,0,0], [0,1,0], [0,0,1]])
    RminusI = R - I3

    # 2. create m initial points and 
    # find curr center (best point)
    print("\nCreating %d initial random points " % m)
    points = np.random.uniform(low=-5.0, high=5.0, size=(m, n))

    center = find_center(points)
    print("\nInitial center (best) point: ")
    print(center)

    # 3. spiral points towards curr center, 
    # update center, repeat
    print("\nUsing spiral dynamics optimization: ")
    for itr in range(max_iter):
        if itr % 100 == 0:
            print("itr = %5d  curr center = " % itr, end="")
            print(center)
        for i in range(m):  # move each pt toward center
            x = points[i]
            points[i] = move_point(x, R, RminusI, center)
        center = find_center(points)  # find new center
  
    # 4. show best center found 
    best_center = find_center(points)
    best_objective = objective_function(best_center)

    print("\nBest center found: ")
    print(best_center)
    print("Best objective value: %0.4f" % best_objective)

    print("\nEnd spiral dynamics optimization demo ")

    print("\nEnd spiral dynamics optimization demo ")  

if __name__ == "__main__":
    main()