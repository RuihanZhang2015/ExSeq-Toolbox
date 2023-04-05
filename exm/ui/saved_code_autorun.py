def do_icp_point(p1,p2,x,y):
    # Ignore Z
    p1[:,2]=0
    p2[:,2]=0

    # Get #num points closest to crosshair
    num = 200
    nbrs = NearestNeighbors(n_neighbors=num, algorithm='ball_tree').fit(p2)
    distances, indices2 = nbrs.kneighbors([[x,y,0]])
    indices2 = indices2[0][distances[0]<20]

    if len(indices2)==0:
        raise ValueError

    # Get closest points to the #num neighbors to the crosshair
    ratio = 0.4
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(p1)
    distances, indices1 = nbrs.kneighbors(p2[indices2,:])
    ds=distances[:,0]
    hist = list()
    for i in range(1,4):
        hist.append(np.count_nonzero(ds<i))

    indices1 = indices1[:,0]

    argsort=ds.argsort()[:int(num*ratio)]
    indices1 = indices1[argsort]
    indices2 = indices2[argsort]

    ds=ds[argsort]
    indices1=indices1[ds<3.0]
    indices2=indices2[ds<3.0]

    if len(indices1)==0:
        raise ValueError


    v = p1[indices1]-p2[indices2]
    icp = np.mean(v, axis=0)
    return icp, np.mean(v*v)


def plot_mesh(img, mesh, color = (255,255,255)):
    for k,v in mesh.items():
        cx,cy = k
        x1,y1 = v
        p2 = mesh.get((cx+1, cy), None)
        p3 = mesh.get((cx, cy+1), None)
        if p2:
            line(img, x1,y1, p2[0], p2[1], color)
        if p3:
            line(img, x1,y1, p3[0], p3[1], color)


def transform_point(mesh, pt, ms):
    orig = np.array(mesh[(0,0)])
    gcoord = (((np.array(pt) - orig) / ms) // 1).astype(int)
    relx,rely = ((np.array(pt) - orig) / ms)%1.0
    print(relx, rely)
    if (gcoord[0], gcoord[1]) not in mesh.keys():
        return None
    if (gcoord[0]+1, gcoord[1]) not in mesh.keys():
        return None
    if (gcoord[0], gcoord[1]+1) not in mesh.keys():
        return None
    if (gcoord[0]+1, gcoord[1]+1) not in mesh.keys():
        return None

    tl = np.array(mesh[(gcoord[0], gcoord[1])])
    tr = np.array(mesh[(gcoord[0]+1, gcoord[1])])
    bl = np.array(mesh[(gcoord[0], gcoord[1]+1)])
    br = np.array(mesh[(gcoord[0]+1, gcoord[1]+1)])

    i1 = (tl*(1-relx) + tr*relx)
    i2 = (bl * (1 - relx) + br * relx)

    return (i1*(1-rely) + i2*rely)


# points transform and filter

rot = float(self.form.sliderR.value())
trans = np.array((float(self.form.sliderX.value()),
                  float(self.form.sliderY.value()),
                  0))
z1 = int(self.form.sliderZ1.value())
z2 = int(self.form.sliderZ2.value())
p1 = filter_points(self.pt1,0,z1)
p2 = filter_points(self.pt2,0,z2)
p1[:,2] = np.max(p1[:,2])-p1[:,2]
p1[:,0] = 150+np.max(p1[:,0])-p1[:,0]
p1 = p1*0.25
p2 = p2*0.25

center_rot = np.array((145,151,0))-trans

r=Rotation.from_euler('z', rot, degrees=True)

p2=p2-center_rot
p2=np.matmul(r.as_matrix(), p2.T).T
p2=p2+center_rot
p2=p2+trans



# Plotting
w,h,t = np.max(np.vstack((p1,p2)), axis=0).astype(np.uint)+1
self.outimg = np.zeros((h,w,3), dtype=np.uint8)
self.outimg2 = np.zeros((t,w,3), dtype=np.uint8)

# Show crosshair
x,y = (self.last_click_x, self.last_click_y)
self.outimg[y-2:y+2, x, :] = 255
self.outimg[y, x-2:x+2, :] = 255

## ICP

# Ignore Z
oldp1=np.copy(p1)
oldp2=np.copy(p2)
p1[:,2]=0
p2[:,2]=0

# Get #num points closest to crosshair
num = 200
nbrs = NearestNeighbors(n_neighbors=num, algorithm='ball_tree').fit(p2)
distances, indices2 = nbrs.kneighbors([[x,y,0]])
indices2 = indices2[0]


ms = 25 # mesh size
mesh = dict()
dif = dict()
visited = set()
to_visit = set()
maxnodes=100

# grid coords, pre-icp pixel coords, icp deformation
to_visit.add((0,0,x,y,0,0))
while len(to_visit)>0 and len(visited)<maxnodes:
    s = sorted(list(to_visit), key=lambda x:x[0]**2+x[1]**2)
    # print(s)
    # gx,gy, px, py, dx, dy = to_visit.pop()
    v = s[0]
    to_visit.remove(v)
    gx,gy, px, py, dx, dy = v
    if (gx, gy) in visited:
        continue
    visited.add((gx, gy))
    # print((gx,gy), (px,py))
    if px<0 or px>=self.outimg.shape[1] or py<0 or py>=self.outimg.shape[0]:
        # print("out")
        continue
    pp2 = p2 + np.array([px, py, 0])
    try:
        dv, err = do_icp_point(p1, p2, px, py)
        if isnan(dv[0]) or isnan(dv[1]):
            raise ValueError
    except ValueError:
        # print("ValueError", len(to_visit))
        continue


    dif[(gx,gy)] = (px+dv[0], py+dv[1])
    for nx,ny in ((1,0), (-1,0), (0,-1), (0,1)):
#    for nx,ny in ((0,1),(1,0)):
        if (gx+nx,gy+ny) not in visited:
            to_visit.add((gx+nx,gy+ny, px+(ms*nx)+dv[0], py+(ms*ny)+dv[1], dx+dv[0], dy+dv[1]))


# Compute transformations

basemesh = dict()
for k in dif.keys():
    basemesh[k]=(x+ms*k[0], y+ms*k[1])


newpoints = list()
for pt in p2:
    newpoint = transform_point(dif, pt[:2], ms)
    if newpoint is not None:
        newpoints.append(newpoint)

# print(len(newpoints), p1.shape)
# mesh[(0,0)]=(x,y)
# mesh[(-1,0)]=(x-ms,y)
# mesh[(1,0)]=(x+ms,y)
# mesh[(0,1)]=(x,y+ms)
# mesh[(0,-1)]=(x,y-ms)
# mesh[(1,1)]=(x+ms,y+ms)
# plot_mesh(self.outimg, mesh, (127,127,127))

# for k,v in list(mesh.items()):
#     px,py = v
#     dv, _ = do_icp_point(p1,pp2, px,py)
#     dif[k] = dv+ np.array([px,py,0])
self.log(dif)


# Ploting

if self.form.showGrids.isChecked():
    plot_mesh(self.outimg, basemesh, (127,127,127))
    plot_mesh(self.outimg, dif, (255,255,255))


if self.form.showOrigP1.isChecked():
    plot_on(p1[:,:2], self.outimg, (255,0,0))
    plot_on(oldp1[:,[0,2]], self.outimg2, (255,0,0))
    
if self.form.showOrigP2.isChecked():
    plot_on(p2[:,:2], self.outimg, (0,255,0))
    plot_on(oldp2[:,[0,2]], self.outimg2, (0,255,0))
    plot_on(p2[indices2[distances[0]<20],:2], self.outimg, (255,255,255))

if self.form.showDefPoints.isChecked():
    plot_on(np.array(newpoints), self.outimg, (0,255,128))
