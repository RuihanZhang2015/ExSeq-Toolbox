# import matplotlib.pyplot as plt
# import h5py
# import plotly.graph_objects as go
# import plotly.express as px


def in_region(self,coord,ROI_min,ROI_max):
        
    """in_region(self,coord,ROI_min,ROI_max)"""

    import numpy as np
    coord = np.asarray(coord)
    ROI_min = np.asarray(ROI_min)
    ROI_max = np.asarray(ROI_max)

    if np.all(coord>=ROI_min) and np.all(coord<ROI_max):
        return True
    else:
        return False
        
# Raw
def inspect_raw_plotly(args,fov,code,c,ROI_min,ROI_max,vmax):
        
    '''
    exseq.inspect_raw_plotly(
                fov=
                ,code=
                ,c=
                ,ROI_min=
                ,ROI_max=
                ,zmax = 600)
    '''
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max
    import plotly.express as px
    
    img = args.retrieve_img(fov,code,c,ROI_min,ROI_max)
    gaussian_filter(img, 1, output=img, mode='reflect', cval=0)

    fig = px.imshow(img, zmax = vmax)
    fig.show()


def inspect_raw_matplotlib(args,fov,code,c,ROI_min,ROI_max,vmax=500):

    '''
        exseq.inspect_raw_matplotlib(
                fov=
                ,code=
                ,c=
                ,ROI_min=
                ,ROI_max=
                ,vmax = 600)
    '''

    import matplotlib.pyplot as plt
    import h5py

    img = args.retrieve_img(fov,code,c,ROI_min,ROI_max)
    
    fig,ax = plt.subplots(1,1)
    ax.imshow(img, vmax)
    plt.show()
        

# Local maximum
def inspect_localmaximum_matplotlib(args,fov,code,ROI_min,ROI_max,vmax):

    import matplotlib.pyplot as plt
    import h5py

    fig,ax = plt.subplots(1,5,figsize = (20,5))
    for c in range(5):
        img = args.retrieve_img(fov,code,c,ROI_min,ROI_max)
        ax[c].imshow(img, vmax = vmax)
    plt.show()


def inspect_localmaximum_plotly(args, fov, code, c, ROI_min, ROI_max):
        
    import plotly.graph_objects as go
    import numpy as np
    import h5py
    import pickle

    fig = go.Figure()

    ## Surface -------------
    for zz in np.linspace(ROI_min[0],ROI_max[0],6):

        img = args.retrieve_img(fov,code,c,[int(zz),ROI_min[1],ROI_min[2]],[int(zz),ROI_max[1],ROI_max[2]])

        y = list(range(ROI_min[1], ROI_max[1]))
        x = list(range(ROI_min[2], ROI_max[2]))
        z = np.ones((ROI_max[1]-ROI_min[1],ROI_max[2]-ROI_min[2])) * (int(zz)+0.2*c)

        fig.add_trace(go.Surface(x=x, y=y, z=z,
            surfacecolor=img,
            cmin=0, 
            cmax=500,
            colorscale=args.colorscales[c],
            showscale=False,
            opacity = 0.2,
        ))

    ## Scatter --------------
    with open(args.project_path +'processed/fov{}/coords_total_code{}.pkl'.format(fov,code), 'rb') as f:
        coords_total = pickle.load(f)
        temp = []
        for coord in coords_total['c{}'.format(c)]:
            if in_region(coord,ROI_min,ROI_max):
                temp.append(coord)     
        temp = np.asarray(temp)
        if len(temp) > 0:
            fig.add_trace(go.Scatter3d(
                z=temp[:,0],
                y=temp[:,1],
                x=temp[:,2],
                mode = 'markers',
                marker = dict(
                    color = args.colors[c],
                    size = 4 ,
                )
            ))

    # ---------------------
    fig.update_layout(
        title="fov{}, code{}, channel {}".format(fov,code,args.channel_names[c]),
        width=800,
        height=800,
        scene=dict(
            aspectmode = 'data',
            xaxis_visible = True,
            yaxis_visible = True, 
            zaxis_visible = True, 
            xaxis_title = "X",
            yaxis_title = "Y",
            zaxis_title = "Z" ,
        ))

    fig.show()


# puncta in ROIs
def inspect_puncta_ROI_matplotlib(args, fov, code, position, centered=40):

    import matplotlib.pyplot as plt
    import numpy as np

    reference = args.retrieve_all_puncta(fov)
        
    fig,axs = plt.subplots(4,10,figsize = (15,7))

    for c in range(4):
        for z_ind,z in enumerate(np.linspace(position[0] - 10,position[0] + 10,10)):
            ROI_min = [int(z),position[1] - centered, position[2] - centered]
            ROI_max = [int(z),position[1] + centered, position[2] + centered]
            img = args.retrieve_img(fov,code,c,ROI_min,ROI_max)
            axs[c,z_ind].imshow(img,cmap=plt.get_cmap(args.colorscales[c]),vmax = 150)

    ROI_min = [position[0] - 10,position[1] - centered, position[2] - centered]
    ROI_max = [position[0] + 10,position[1] + centered, position[2] + centered]    
    temp = [x['code{}'.format(code)] for x in reference if 'code{}'.format(code) in x and in_region(x['code{}'.format(code)]['position'], ROI_min,ROI_max) ] 

    for c in range(4):
        temp2 = [x['c{}'.format(c)]['position'] for x in temp if 'c{}'.format(c) in x]
        for puncta in temp2:
            axs[c,(puncta[0]-position[0]+10)//2].scatter(puncta[1]-position[1]+centered,puncta[2]-position[2]+centered,s = 20)

    fig.suptitle('fov{} code{}'.format(fov,code))    
    plt.show()
        

def inspect_puncta_ROI_plotly(args, fov, codes, position, c_list=[0,1,2,3], centered=40):

    import plotly.graph_objects as go
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt

    spacer = 40
    ROI_min, ROI_max = [position[0]-centered:position[0]+centered,position[1]-centered:position[1]+centered,position[2]-centered:position[2]+centered]
    reference = args.retrieve_result(fov)

    fig = go.Figure()

    for i,code in enumerate(codes):

        ## Surface -------------
        for c in c_list:

            for zz in np.linspace(ROI_min[0],ROI_max[0],7):

                img = args.retrieve_img(fov,code,c,[int(zz),ROI_min[1],ROI_min[2]],[int(zz),ROI_max[1],ROI_max[2]])

                y = list(range(ROI_min[1], ROI_max[1]))
                x = list(range(ROI_min[2], ROI_max[2]))
                z = np.ones((ROI_max[1]-ROI_min[1],ROI_max[2]-ROI_min[2])) * ( int(zz)+0.7*c+i*spacer )
                fig.add_trace(go.Surface(x=x, y=y, z=z,
                            surfacecolor=img,
                            cmin=0, 
                            cmax=500,
                            colorscale=args.colorscales[c],
                            showscale=False,
                            opacity = 0.2,
                        ))

        ## Scatter --------------
        temp = [x['code{}'.format(code)] for x in reference if 'code{}'.format(code) in x and in_region(x['code{}'.format(code)]['position'], ROI_min,ROI_max) ] 

        for c in c_list:

                temp2 = [x['c{}'.format(c)]['position'] for x in temp if 'c{}'.format(c) in x]
                temp2 = np.asarray(temp2)
                if len(temp2)==0:
                    continue
                fig.add_trace(go.Scatter3d(
                        z=temp2[:,0] + i*spacer,
                        y=temp2[:,1],
                        x=temp2[:,2],
                        mode = 'markers',
                        marker = dict(
                            color = args.colors[c],
                            size=4,
                        )
                    ))

    # ------------
    fig.add_trace(go.Scatter3d(
                    z= [ROI_min[0], ROI_max[0] + (len(codes)-1)*spacer],
                    y= [(ROI_min[1]+ROI_max[1])/2]*2,
                    x= [(ROI_min[2]+ROI_max[2])/2]*2,
                    mode = 'lines',
                    line = dict(
                        color = 'black',
                        width = 10,
                    )
                ))        


    # ---------------------
    fig.update_layout(
        title = "Inspect fov{}, code ".format(fov) + 'and '.join([str(x) for x in codes]),
        width = 800,
        height = 800,

        scene=dict(
            aspectmode = 'data',
            xaxis_visible=True,
            yaxis_visible=True, 
            zaxis_visible=True, 
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z" ,
        ))

    fig.show()


# Individual puncta
def inspect_puncta_individual_matplotlib(args,fov,puncta_index, centered=40):

    import matplotlib.pyplot as plt
    puncta = args.retrieve_puncta(fov,puncta_index)
        
    fig,axs = plt.subplots(4,len(args.codes),figsize = (15,7))
            
    for code_ind,code in enumerate(args.codes):
            
            if 'code{}'.format(code) not in puncta:
                continue
                
            position = puncta['code{}'.format(code)]['position']
            ROI_min = [int(position[0]),position[1] - centered, position[2] - centered]
            ROI_max = [int(position[0]),position[1] + centered, position[2] + centered]
            for c in range(4):
                img = args.retrieve_img(fov,code,c,ROI_min,ROI_max)
                axs[c,code_ind].imshow(img,cmap=plt.get_cmap(args.colorscales[c]),vmax = 150)
                axs[c,code_ind].set_title('{0:0.2f}'.format(img[centered,centered]))
            
            axs[puncta['code{}'.format(code)]['color'],code_ind].scatter(centered,centered,c = 'white')

        fig.suptitle('fov{} puncta{}'.format(fov,puncta_index))    
        plt.show() 
   
        
def inspect_puncta_individual_plotly(args, fov, puncta_index,spacer = 40 ):

        '''
        exseq.inspect_puncta(
                fov = 
                ,puncta_index = 
                ,spacer = 40 
                )
        '''

        codes = self.args.codes

        puncta = self.retrieve_puncta(fov,puncta_index)

        
        fig = go.Figure()
        for i, code in enumerate(codes):

            if 'code{}'.format(code) in interest:

                print('code{}'.format(code))
                puncta = reference[puncta_index]['code{}'.format(code)]   
                d0, d1, d2 = puncta['position']
                ROI_min = [d0-10, d1-40, d2-40]
                ROI_max = [d0+10, d1+40, d2+40]

                print('ROI_min = [{},{},{}]'.format(*ROI_min))
                print('ROI_max = [{},{},{}]'.format(*ROI_max))
                # pprint.pprint(puncta)


                c_candidates = []

                ## Surface -------------
                for c in range(4):

                    if 'c{}'.format(c) in puncta:

                        c_candidates.append(c)

                        for zz in np.linspace(ROI_min[0],ROI_max[0],7):

                            with h5py.File(self.args.h5_path.format(code,fov), "r") as f:
                                im = f[self.args.channel_names[c]][int(zz),ROI_min[1]:ROI_max[1],ROI_min[2]:ROI_max[2]]
                                im = np.squeeze(im)
                            y = list(range(ROI_min[1], ROI_max[1]))
                            x = list(range(ROI_min[2], ROI_max[2]))
                            z = np.ones((ROI_max[1]-ROI_min[1],ROI_max[2]-ROI_min[2])) * (int(zz)+0.5*c + i* spacer)
                            fig.add_trace(go.Surface(x=x, y=y, z=z,
                                surfacecolor=im,
                                cmin=0, 
                                cmax=500,
                                colorscale=self.args.colorscales[c],
                                showscale=False,
                                opacity = 0.2,
                            ))

                ## Scatter --------------

                temp = [x['code{}'.format(code)] for x in reference if 'code{}'.format(code) in x and self.in_region(x['code{}'.format(code)]['position'], ROI_min,ROI_max) ] 

                for c in c_candidates:

                    fig.add_trace(go.Scatter3d(
                            z = [puncta['c{}'.format(c)]['position'][0]+ i * spacer], 
                            y = [puncta['c{}'.format(c)]['position'][1]],
                            x = [puncta['c{}'.format(c)]['position'][2]],
                            mode = 'markers',
                            marker = dict(
                                color = 'gray',
                                size= 8,
                                symbol = 'circle-open'
                            )
                        ))

                    temp2 = np.asarray([x['c{}'.format(c)]['position'] for x in temp if 'c{}'.format(c) in x])

                    if len(temp2) == 0:
                        continue

                    fig.add_trace(go.Scatter3d(
                            z = temp2[:,0] + i* spacer,
                            y = temp2[:,1],
                            x = temp2[:,2],
                            mode = 'markers',
                            marker = dict(
                                color = self.args.colors[c],
                                size=4,
                            )
                        ))

        # ---------------------
        fig.update_layout(
            title="puncta index {}".format(puncta_index),
            width=800,
            height=800,

            scene=dict(
                aspectmode = 'data',
                xaxis_visible = True,
                yaxis_visible = True, 
                zaxis_visible = True, 
                xaxis_title = "X",
                yaxis_title = "Y",
                zaxis_title = "Z" ,
            ))

        fig.show()


# Puncta across rounds
def inspect_between_rounds(args, fov, code1, code2, ROI_min, ROI_max):

        '''
        exseq.inspect_fov_all_to_all(
                fov=
                ,code1=
                ,code2=
                ,ROI_min=
                ,ROI_max=
                )
        '''

        spacer = 100

        with open(self.args.work_path +'/fov{}/result.pkl'.format(fov), 'rb') as f:
            reference = pickle.load(f)
        reference = [ x for x in reference if self.in_region(x['position'], ROI_min,ROI_max) ] 

        fig = go.Figure()

        ## Lines ====================

        temp = [x for x in reference if (code1 in x) and (code2 in x) ]
        for x in temp:
            center1,center2 = x[code1]['position'], x[code2]['position']
            name = x['index']
            fig.add_trace(go.Scatter3d(
                z=[center1[0],center2[0]+spacer],
                y=[center1[1],center2[1]],
                x=[center1[2],center2[2]],
                mode = 'lines',
                name = name,
                line = dict(
                    color = 'gray',
                    # size=4,
                )
            ))
            

        ## Code1  =========================

        temp = [x for x in reference if (code1 in x)]


        ### Centers
        points = [x[code1]['position'] for x in temp]
        points = np.asarray(points)
        texts = [x['index'] for x in temp]
        if len(points)>0:
            fig.add_trace(go.Scatter3d(
                    z=points[:,0],
                    y=points[:,1],
                    x=points[:,2],
                    text = texts,
                    mode = 'markers+text',
                    name = 'consensus',
                    marker = dict(
                        color = 'gray',
                        size=10,
                        opacity = 0.2,
                    )
                ))

        ## Scatters --------------

        for c in range(4):

            points = [x[code1]['c{}'.format(c)]['position'] for x in temp if 'c{}'.format(c) in x[code1]]
            points = np.asarray(points)
            if len(points) == 0:
                continue

            fig.add_trace(go.Scatter3d(
                z=points[:,0],
                y=points[:,1],
                x=points[:,2],
                name = 'channels',
                mode = 'markers',
                marker = dict(
                    color = self.args.colors[c],
                    size=4,
                )
            ))

        ## Lines --------------

        for x in temp:
            points = [ x[code1][c]['position'] for c in ['c0','c1','c2','c3'] if c in x[code1] ]

            for i in range(len(points)-1):
                for j in range(i+1,len(points)):

                    fig.add_trace(go.Scatter3d(
                        z = [ points[i][0], points[j][0] ],
                        y = [ points[i][1], points[j][1] ],
                        x = [ points[i][2], points[j][2] ],
                        mode = 'lines',
                        name = 'inter channel',
                        line = dict(
                            color = 'gray',
                            # size=4,
                        )
                    ))   


        ## Code2  =========================


        temp = [x for x in reference if (code2 in x)]

        ### Centers
        points = [x[code2]['position'] for x in temp]
        points = np.asarray(points)
        texts = [x['index'] for x in temp]

        if len(points)>0:
            fig.add_trace(go.Scatter3d(
                z=points[:,0] + spacer,
                y=points[:,1],
                x=points[:,2],
                text = texts,
                mode = 'markers+text',
                name = 'consensus',
                marker = dict(
                    color = 'gray',
                    size=10,
                    opacity = 0.2,
                )
            ))

        ## Scatters --------------

        for c in range(4):

            points = [x[code2]['c{}'.format(c)]['position'] for x in temp if 'c{}'.format(c) in x[code2]]
            points = np.asarray(points)
            if len(points) == 0:
                continue
            fig.add_trace(go.Scatter3d(
                z=points[:,0] + spacer,
                y=points[:,1],
                x=points[:,2],
                mode = 'markers',
                name = 'channels',
                marker = dict(
                    color = self.args.colors[c],
                    size=4,
                )
            ))

        ## Lines --------------

        for x in temp:
            points = [ x[code2][c]['position'] for c in ['c0','c1','c2','c3'] if c in x[code2] ]
            for i in range(len(points)-1):
                for j in range(i+1,len(points)):
                    fig.add_trace(go.Scatter3d(
                        z = [ points[i][0]+spacer, points[j][0]+spacer ],
                        y = [ points[i][1], points[j][1] ],
                        x = [ points[i][2], points[j][2] ],
                        mode = 'lines',
                        name = 'inter channel',
                        line = dict(
                            color = 'gray',
                            # size=4,
                        )
                    ))        

        # ---------------------
        fig.update_layout(
            title="My 3D scatter plot",
            width=800,
            height=800,
            showlegend=False,
            scene=dict(
                aspectmode = 'data',
                xaxis_visible=True,
                yaxis_visible=True, 
                zaxis_visible=True, 
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z" ,
            ))

        fig.show()


def inspect_across_rounds(args, fov, ROI_min, ROI_max):

        '''
        exseq.inspect_fov_all(
                fov=
                ,ROI_min=
                ,ROI_max=
                )
        '''

        spacer = 100

        with open(self.args.work_path +'/fov{}/result.pkl'.format(fov), 'rb') as f:
            reference = pickle.load(f)
        reference = [ x for x in reference if self.in_region(x['position'], ROI_min,ROI_max) ] 
#         print(reference)

        fig = go.Figure()

        ## Lines ====================

        for i1 in range(len(self.args.codes))[:-1]:

            code1 = 'code{}'.format(self.args.codes[i1])
            i2 = i1+1
            code2 = 'code{}'.format(self.args.codes[i2])

            temp = [x for x in reference if (code1 in x) and (code2 in x) ]
            for x in temp:
                    center1,center2 = x[code1]['position'], x[code2]['position']
                    name = x['index']
                    fig.add_trace(go.Scatter3d(
                        z=[center1[0]+i1*spacer,center2[0]+i2*spacer],
                        y=[center1[1],center2[1]],
                        x=[center1[2],center2[2]],
                        mode = 'lines',
                        name = name,
                        line = dict(
                            color = 'gray',
                            # size=4,
                        )
                    ))


        ## Code1  =========================

        for ii,code in enumerate(self.args.codes):

            code1 = 'code{}'.format(code)

            temp = [x for x in reference if (code1 in x)]

            ### Centers
            points = [x[code1]['position'] for x in temp]
            points = np.asarray(points)
            texts = [x['index'] for x in temp]
            if len(points)>0:
                fig.add_trace(go.Scatter3d(
                        z=points[:,0]+ii*spacer,
                        y=points[:,1],
                        x=points[:,2],
                        text = texts,
                        mode = 'markers+text',
                        name = 'consensus',
                        marker = dict(
                            color = 'gray',
                            size=10,
                            opacity = 0.2,
                        )
                    ))

            ## Scatters --------------

            for c in range(4):

                points = [x[code1]['c{}'.format(c)]['position'] for x in temp if 'c{}'.format(c) in x[code1]]
                points = np.asarray(points)
                if len(points) == 0:
                    continue

                fig.add_trace(go.Scatter3d(
                    z=points[:,0]+ii*spacer,
                    y=points[:,1],
                    x=points[:,2],
                    name = 'channels',
                    mode = 'markers',
                    marker = dict(
                        color = self.args.colors[c],
                        size=4,
                    )
                ))

            ## Lines --------------

            for x in temp:
                points = [ x[code1][c]['position'] for c in ['c0','c1','c2','c3'] if c in x[code1] ]

                for i in range(len(points)-1):
                    for j in range(i+1,len(points)):

                        fig.add_trace(go.Scatter3d(
                            z = [ points[i][0]+ii*spacer, points[j][0]+ii*spacer ],
                            y = [ points[i][1], points[j][1] ],
                            x = [ points[i][2], points[j][2] ],
                            mode = 'lines',
                            name = 'inter channel',
                            line = dict(
                                color = 'gray',
                                # size=4,
                            )
                        ))   


        # ---------------------
        fig.update_layout(
            title="My 3D scatter plot",
            width=800,
            height=800,
            showlegend=False,
            scene=dict(
                aspectmode = 'data',
                xaxis_visible=True,
                yaxis_visible=True, 
                zaxis_visible=True, 
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z" ,
            ))

        fig.show()
    
