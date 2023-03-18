
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from exm.utils import retrieve_img,retrieve_all_puncta,retrieve_one_puncta

import plotly.graph_objects as go
import plotly.express as px


def in_region(coord,ROI_min,ROI_max):
        
    """in_region(self,coord,ROI_min,ROI_max)"""

    coord = np.asarray(coord)
    ROI_min = np.asarray(ROI_min)
    ROI_max = np.asarray(ROI_max)

    if np.all(coord>=ROI_min) and np.all(coord<ROI_max):
        return True
    else:
        return False
        
# Raw plotly
def inspect_raw_plotly(args,fov,code,channel,ROI_min,ROI_max,vmax=500,mode ='raw'):
        
    img = retrieve_img(args,fov,code,channel,ROI_min,ROI_max)
    if mode == 'blur':
        gaussian_filter(img, 1, output=img, mode='reflect', cval=0)

    fig = px.imshow(img, zmax = vmax,title='Raw Fov {} Code {} Channel {}'.format(fov,code,args.channel_names[channel]),labels=dict(color="Intensity"))
    fig.show()

# raw matplotlib
def inspect_raw_matplotlib(args,fov,code,channel,ROI_min,ROI_max,vmax=500,mode = 'raw'):

    '''
        exseq.inspect_raw_matplotlib(
                fov=
                ,code=
                ,channel=
                ,ROI_min=
                ,ROI_max=
                ,vmax = 600)
    '''

    img = retrieve_img(args,fov,code,channel,ROI_min,ROI_max)
    if mode == 'blur':
        gaussian_filter(img, 1, output=img, mode='reflect', cval=0)

    fig,ax = plt.subplots(1,1)
    ax.set_title('Raw Fov {} Code {} Channel {}'.format(fov,code,args.channel_names[channel]))
    im = ax.imshow(img, vmax=vmax)
    cbar = fig.colorbar(im)
    cbar.set_label('Intensity')
    plt.show()
        

# Local maximum matplotlib
def inspect_localmaximum_matplotlib(args,fov,code,ROI_min,ROI_max,vmax=500):

    fig,ax = plt.subplots(1,5,figsize = (20,5))
    for channel in range(5):
        img = retrieve_img(args,fov,code,channel,ROI_min,ROI_max)
        ax[channel].imshow(img, vmax = vmax)
        ax[channel].set_title('Channel {}'.format(args.channel_names[channel]))
    fig.suptitle('local maximum Fov {} Code {}'.format(fov,code), fontsize=16)
    plt.show()

# Local maximum plotly
def inspect_localmaximum_plotly(args, fov, code, channel, ROI_min, ROI_max):
        
    fig = go.Figure()

    ## Surface -------------
    for zz in np.linspace(ROI_min[0],ROI_max[0],6):

        img = retrieve_img(args,fov,code,channel,[int(zz),ROI_min[1],ROI_min[2]],[int(zz),ROI_max[1],ROI_max[2]])

        y = list(range(ROI_min[1], ROI_max[1]))
        x = list(range(ROI_min[2], ROI_max[2]))
        z = np.ones((ROI_max[1]-ROI_min[1],ROI_max[2]-ROI_min[2])) * (int(zz)+0.2*channel)

        fig.add_trace(go.Surface(x=x, y=y, z=z,
            surfacecolor=img,
            cmin=0, 
            cmax=500,
            colorscale=args.colorscales[channel],
            showscale=False,
            opacity = 0.2,
        ))

    ## Scatter --------------
    with open(args.work_path +'/fov{}/coords_total_code{}.pkl'.format(fov,code), 'rb') as f:
        coords_total = pickle.load(f)
        temp = []
        for coord in coords_total['c{}'.format(channel)]:
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
                    color = args.colors[channel],
                    size = 4 ,
                )
            ))

    # ---------------------
    fig.update_layout(
        title="fov{}, code{}, channel {}".format(fov,code,args.channel_names[channel]),
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
def inspect_puncta_ROI_matplotlib(args, fov, code, position,center_dist=40):

    reference = retrieve_all_puncta(args,fov)
        
    fig,axs = plt.subplots(4,10,figsize = (15,7),dpi=100)

    for channel in range(4):
        for z_ind,z in enumerate(np.linspace(position[0] - 10,position[0] + 10,10)):
            ROI_min = [int(z),position[1] - center_dist, position[2] - center_dist]
            ROI_max = [int(z),position[1] + center_dist, position[2] + center_dist]
            img = retrieve_img(args,fov,code,channel,ROI_min,ROI_max)
            axs[channel,z_ind].imshow(img,cmap=plt.get_cmap(args.colorscales[channel]),vmax = 150)
            if channel == 3:
                axs[channel,z_ind].set_xlabel('{0:0.0f}'.format(z))
            if z_ind == 0:
                axs[channel,z_ind].set_ylabel(args.channel_names[channel])

    ROI_min = [position[0] - 10,position[1] - center_dist, position[2] - center_dist]
    ROI_max = [position[0] + 10,position[1] + center_dist, position[2] + center_dist]    
    temp = [x['code{}'.format(code)] for x in reference if 'code{}'.format(code) in x and in_region( x['code{}'.format(code)]['position'], ROI_min,ROI_max) ] 

    for channel in range(4):
        temp2 = [x['c{}'.format(channel)]['position'] for x in temp if 'c{}'.format(channel) in x]
        for puncta in temp2:
            axs[channel,(puncta[0]-position[0]+10)//2].scatter(puncta[1]-position[1]+center_dist,puncta[2]-position[2]+center_dist, marker = 'o', s = 30)

    fig.suptitle('Fov {} Code {}'.format(fov,code))    
    plt.show()
        

def inspect_puncta_ROI_plotly(args, fov, position, c_list = [0,1,2,3],center_dist=40,spacer=40):

    

    ROI_min = [position[0]-10,position[1]-center_dist,position[2]-center_dist]
    ROI_max = [position[0]+10,position[1]+center_dist,position[2]+center_dist]
    reference = retrieve_all_puncta(args,fov)

    fig = go.Figure()

    for i,code in enumerate(args.codes):

        ## Surface -------------
        for c in c_list:

            for zz in np.linspace(ROI_min[0],ROI_max[0],7):

                img = retrieve_img(args,fov,code,c,[int(zz),ROI_min[1],ROI_min[2]],[int(zz),ROI_max[1],ROI_max[2]]) 

                y = list(range(ROI_min[1], ROI_max[1]))
                x = list(range(ROI_min[2], ROI_max[2]))
                z = np.ones((ROI_max[1]-ROI_min[1],ROI_max[2]-ROI_min[2])) * (int(zz)+0.7*c+i*spacer)
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
                    z= [ROI_min[0], ROI_max[0] + (len(args.codes)-1)*spacer],
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
        title = "Inspect fov{}, code: ".format(fov) + ' '.join([str(x) for x in args.codes]),
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
def inspect_puncta_individual_matplotlib(args, fov, puncta_index,center_dist = 40):

    import matplotlib.pyplot as plt
    puncta = retrieve_one_puncta(args,fov,puncta_index) 

    fig,axs = plt.subplots(4,len(args.codes),figsize = (15,11))
            
    for code_ind,code in enumerate(args.codes):
            
        if 'code{}'.format(code) not in puncta:
            for c in range(4):
                fig.delaxes(axs[c,code_ind])
            continue
                
        position = puncta['code{}'.format(code)]['position']
        ROI_min = [int(position[0]),position[1] - center_dist, position[2] - center_dist]
        ROI_max = [int(position[0]),position[1] + center_dist, position[2] + center_dist]
        for c in range(4):
            img = retrieve_img(args,fov,code,c,ROI_min,ROI_max)
            axs[c,code_ind].imshow(img,cmap=plt.get_cmap(args.colorscales[c]),vmax = 150)
            axs[c,code_ind].set_title('code{}'.format(code))
            axs[c,code_ind].set_xlabel('{0:0.2f}'.format(img[center_dist,center_dist]))
            axs[c,code_ind].set_ylabel(args.channel_names[c])
            
        axs[puncta['code{}'.format(code)]['color'],code_ind].scatter(center_dist,center_dist,c = 'white')

    fig.suptitle('fov{} puncta{}'.format(fov,puncta_index))    
    fig.tight_layout()
    plt.show() 
    
        
def inspect_puncta_individual_plotly(args, fov, puncta_index,center_dist=40,spacer = 40):

    reference = retrieve_all_puncta(args,fov)
    puncta = retrieve_one_puncta(args,fov, puncta_index)

    fig = go.Figure()
    for i, code in enumerate(args.codes):

        if 'code{}'.format(code) in puncta:

            print('code{}'.format(code))
            puncta = puncta['code{}'.format(code)]   
            d0, d1, d2 = puncta['position']
            ROI_min = [d0-10, d1-center_dist, d2-center_dist]
            ROI_max = [d0+10, d1+center_dist, d2+center_dist]

            print('ROI_min = [{},{},{}]'.format(*ROI_min))
            print('ROI_max = [{},{},{}]'.format(*ROI_max))

            c_candidates = []

            ## Surface -------------
            for c in range(4):

                if 'c{}'.format(c) in puncta:

                    c_candidates.append(c)

                    for zz in np.linspace(ROI_min[0],ROI_max[0],7):

                        with h5py.File(args.h5_path.format(code,fov), "r") as f:
                            im = f[args.channel_names[c]][int(zz),ROI_min[1]:ROI_max[1],ROI_min[2]:ROI_max[2]]
                            im = np.squeeze(im)
                        y = list(range(ROI_min[1], ROI_max[1]))
                        x = list(range(ROI_min[2], ROI_max[2]))
                        z = np.ones((ROI_max[1]-ROI_min[1],ROI_max[2]-ROI_min[2])) * (int(zz)+0.5*c + i* spacer)
                        fig.add_trace(go.Surface(x=x, y=y, z=z,
                            surfacecolor=im,
                            cmin=0, 
                            cmax=500,
                            colorscale=args.colorscales[c],
                            showscale=False,
                            opacity = 0.2,
                        ))

            ## Scatter --------------

            temp = [x['code{}'.format(code)] for x in reference if 'code{}'.format(code) in x and in_region(x['code{}'.format(code)]['position'], ROI_min,ROI_max) ] 

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
                            color = args.colors[c],
                            size=4,
                        )
                    ))

    # ---------------------
    fig.update_layout(
        title="Puncta Fov{} index {}".format(fov,puncta_index),
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
def inspect_between_rounds_plotly(args, fov, code1, code2, ROI_min, ROI_max,spacer = 40):

    if ROI_max[0]-ROI_min[0]>20:
        print('ROI_max[0]-ROI_min[0]should be smaller than 20')
        return 
    
    code1, code2 = 'code{}'.format(code1),'code{}'.format(code2)

    reference = retrieve_all_puncta(args,fov)
    reference = [ x for x in reference if in_region(x['position'], ROI_min,ROI_max) ] 
    print('Only {} puncta remained'.format(len(reference)))

    fig = go.Figure()

    # Lines  between codes ====================
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
            )
        ))
            

    # Code1  =========================
    temp = [x for x in reference if (code1 in x)]

    # Centers
    points = [x[code1]['position'] for x in temp]
    points = np.asarray(points)
    texts = ['{} {}'.format(x['index'],code1) for x in temp]
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

    # Scatters --------------
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
                color = args.colors[c],
                size=4,
            )
        ))

    # Lines --------------
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
                    )
                ))   


    # Code2  =========================
    temp = [x for x in reference if (code2 in x)]

    # Centers
    points = [x[code2]['position'] for x in temp]
    points = np.asarray(points)
    texts = ['{} {}'.format(x['index'],code2) for x in temp]

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

    # Scatters --------------
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
                color = args.colors[c],
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
            title="Puncta Between Rounds: Fov{} - {} & {}".format(fov,code1,code2),
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


def inspect_across_rounds_plotly(args, fov, ROI_min, ROI_max,spacer = 20):


    reference = retrieve_all_puncta(args,fov)
    reference = [ x for x in reference if in_region(x['position'], ROI_min,ROI_max) ] 

    fig = go.Figure()

    ## Lines ====================
    for puncta in reference:
        codes = sorted([x for x in puncta if x.startswith('code')])
        for i in range(len(codes)-1):
            code1 = codes[i]
            code2 = codes[i+1]

            center1,center2 = puncta[code1]['position'], puncta[code2]['position']
            name = puncta['index']
            fig.add_trace(go.Scatter3d(
                    z=[center1[0]+int(code1[-1])*spacer,center2[0]+int(code2[-1])*spacer],
                    y=[center1[1],center2[1]],
                    x=[center1[2],center2[2]],
                    mode = 'lines',
                    name = name,
                    line = dict(
                        color = 'gray',
                    )
                ))


    ## Code  =========================
    for code in args.codes:

        code_str = 'code{}'.format(code)

        temp = [x for x in reference if (code_str in x)]

        ## Centers
        points = [x[code_str]['position'] for x in temp]
        points = np.asarray(points)
        texts = ['{} {}'.format(x['index'],code_str) for x in temp]
        if len(points)>0:
            fig.add_trace(go.Scatter3d(
                        z=points[:,0]+code*spacer,
                        y=points[:,1],
                        x=points[:,2],
                        text = texts,
                        mode = 'markers+text',
                        name = 'consensus',
                        marker = dict(
                            color = 'gray',
                            size = 10,
                            opacity = 0.2,
                        )
                    ))

        ## Scatters --------------
        for c in range(4):
            points = [x[code_str]['c{}'.format(c)]['position'] for x in temp if 'c{}'.format(c) in x[code_str]]
            points = np.asarray(points)
            if len(points) == 0:
                continue

            fig.add_trace(go.Scatter3d(
                z=points[:,0]+code*spacer,
                y=points[:,1],
                x=points[:,2],
                name = 'channels',
                mode = 'markers',
                marker = dict(
                    color = args.colors[c],
                    size=4,
                )
            ))

        ## Lines --------------
        for x in temp:
            points = [ x[code_str][c]['position'] for c in ['c0','c1','c2','c3'] if c in x[code_str] ]

            for i in range(len(points)-1):
                for j in range(i+1,len(points)):

                    fig.add_trace(go.Scatter3d(
                        z = [ points[i][0]+code*spacer, points[j][0]+code*spacer ],
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
        title="Puncta Across Rounds: FOV{}".format(fov),
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
    
