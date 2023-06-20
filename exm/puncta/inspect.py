"""
Manually check identification and consolidation of puncta.
"""

import os
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from exm.utils import retrieve_img, retrieve_all_puncta, retrieve_one_puncta
from exm.puncta.improve import puncta_all_nearest_points

import plotly.graph_objects as go
import plotly.express as px


def in_region(coord, ROI_min, ROI_max):
    r"""Given a coordinate location and lower and upper bounds for a volume chunk (region), returns whether or not the coordinate is inside the chunk.

    :param list coord: coordinate list, in the format of :math:`[z, y, x]`.
    :param list ROI_min: minimum coordinates of the volume chunk. Expects coordinates in the format of :math:`[z, y, x]`.
    :param list ROI_max: maximum coordinates of the volume chunk. Expects coordinates in the format of :math:`[z, y, x]`.
    """
    coord = np.asarray(coord)
    ROI_min = np.asarray(ROI_min)
    ROI_max = np.asarray(ROI_max)

    if np.all(coord >= ROI_min) and np.all(coord < ROI_max):
        return True
    else:
        return False


# Raw plotly
def inspect_raw_plotly(
    args,
    fov,
    code,
    channel,
    ROI_min,
    ROI_max,
    vmax=500,
    mode="raw",
    export_file_name=False,
):
    r"""Plots the middle slice of a specified volume chunk using Plotly.

    :param args.Args args: configuration options.
    :param int fov: the field of fiew of the volume chunk to be returned.
    :param int code: the code of the volume chunk to be returned.
    :param int channel: the channel of the volume chunk to be returned.
    :param list ROI_min: minimum coordinates of the volume chunk. Expects coordinates in the format of :math:`[z, y, x]`.
    :param list ROI_max: maximum coordinates of the volume chunk. Expects coordinates in the format of :math:`[z, y, x]`.
    :param int vmax: maximum pixel intensity to display. Default: ``500``
    :param str mode: expects 'raw' or 'blur'. 'raw' plots the images as-is, 'blur' applies Gaussian blurring before plotting. Default: ``'raw'``
    :param str export_file_name: name of the file to be exported. Default: ``False``
    """

    img = retrieve_img(args, fov, code, channel, ROI_min, ROI_max)
    if mode == "blur":
        gaussian_filter(img, 1, output=img, mode="reflect", cval=0)

    fig = px.imshow(
        img,
        zmax=vmax,
        title="Raw Fov {} Code {} Channel {}".format(
            fov, code, args.channel_names[channel]
        ),
        labels=dict(color="Intensity"),
    )

    if export_file_name != None:
        fig.write_html(
            os.path.join(
                args.work_path,
                "inspect_puncta/{}".format(str(export_file_name) + ".html"),
            )
        )

    fig.show()


# raw matplotlib
def inspect_raw_matplotlib(
    args, fov, code, channel, ROI_min, ROI_max, vmax=500, mode="raw"
):
    r"""Plots the middle slice of a specified volume chunk using Matplotlib.

    :param args.Args args: configuration options.
    :param int fov: the field of fiew of the volume chunk to be returned.
    :param int code: the code of the volume chunk to be returned.
    :param int channel: the channel of the volume chunk to be returned.
    :param list ROI_min: minimum coordinates of the volume chunk. Expects coordinates in the format of :math:`[z, y, x]`.
    :param lsit ROI_max: maximum coordinates of the volume chunk. Expects coordinates in the format of :math:`[z, y, x]`.
    :param int vmax: maximum pixel intensity to display. Default: ``500``
    :param str mode: expects 'raw' or 'blur'. 'raw' plots the images as-is, 'blur' applies Gaussian blurring before plotting. Default: ``'raw'``
    """

    img = retrieve_img(args, fov, code, channel, ROI_min, ROI_max)
    if mode == "blur":
        gaussian_filter(img, 1, output=img, mode="reflect", cval=0)

    fig, ax = plt.subplots(1, 1)
    ax.set_title(
        "Raw Fov {} Code {} Channel {}".format(fov, code, args.channel_names[channel])
    )
    im = ax.imshow(img, vmax=vmax)
    cbar = fig.colorbar(im)
    cbar.set_label("Intensity")
    plt.show()


# Local maximum matplotlib
def inspect_localmaximum_matplotlib(args, fov, code, ROI_min, ROI_max, vmax=500):
    r"""Plots middle slice of each channel for a specific fov/code using Matplotlib.

    :param args.Args args: configuration options.
    :param int fov: the field of fiew of the volume chunk to be returned.
    :param int code: the code of the volume chunk to be returned.
    :param int channel: the channel of the volume chunk to be returned.
    :param list ROI_min: minimum coordinates of the volume chunk. Expects coordinates in the format of :math:`[z, y, x]`.
    :param list ROI_max: maximum coordinates of the volume chunk. Expects coordinates in the format of :math:`[z, y, x]`.
    :param int vmax: maximum pixel intensity to display. Default: ``500``
    """

    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    for channel in range(5):
        img = retrieve_img(args, fov, code, channel, ROI_min, ROI_max)
        ax[channel].imshow(img, vmax=vmax)
        ax[channel].set_title("Channel {}".format(args.channel_names[channel]))
    fig.suptitle("local maximum Fov {} Code {}".format(fov, code), fontsize=16)
    plt.show()


# Local maximum plotly
def inspect_localmaximum_plotly(
    args, fov, code, channel, ROI_min, ROI_max, export_file_name=None
):
    """Plots identified puncta for a specific fov/code/channel using Plotly.

    :param args.Args args: configuration options.
    :param int fov: the field of fiew of the volume chunk to be returned.
    :param int code: the code of the volume chunk to be returned.
    :param int channel: the channel of the volume chunk to be returned.
    :param list ROI_min: minimum coordinates of the volume chunk. Expects coordinates in the format of :math:`[z, y, x]`.
    :param list ROI_max: maximum coordinates of the volume chunk. Expects coordinates in the format of :math:`[z, y, x]`.
    :param str export_file_name: name of the file to be exported. Default: ``None``
    """

    fig = go.Figure()

    ## Surface -------------
    for zz in np.linspace(ROI_min[0], ROI_max[0], 6):

        img = retrieve_img(
            args,
            fov,
            code,
            channel,
            [int(zz), ROI_min[1], ROI_min[2]],
            [int(zz), ROI_max[1], ROI_max[2]],
        )

        y = list(range(ROI_min[1], ROI_max[1]))
        x = list(range(ROI_min[2], ROI_max[2]))
        z = np.ones((ROI_max[1] - ROI_min[1], ROI_max[2] - ROI_min[2])) * (
            int(zz) + 0.2 * channel
        )

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=img,
                cmin=0,
                cmax=500,
                colorscale=args.colorscales[channel],
                showscale=False,
                opacity=0.2,
            )
        )

    ## Scatter --------------
    with open(
        args.work_path + "/fov{}/coords_total_code{}.pkl".format(fov, code), "rb"
    ) as f:
        coords_total = pickle.load(f)
        temp = []
        for coord in coords_total["c{}".format(channel)]:
            if in_region(coord, ROI_min, ROI_max):
                temp.append(coord)
        temp = np.asarray(temp)
        if len(temp) > 0:
            fig.add_trace(
                go.Scatter3d(
                    z=temp[:, 0],
                    y=temp[:, 1],
                    x=temp[:, 2],
                    mode="markers",
                    marker=dict(
                        color=args.colors[channel],
                        size=4,
                    ),
                )
            )

    # ---------------------
    fig.update_layout(
        title="fov{}, code{}, channel {}".format(
            fov, code, args.channel_names[channel]
        ),
        width=800,
        height=800,
        scene=dict(
            aspectmode="data",
            xaxis_visible=True,
            yaxis_visible=True,
            zaxis_visible=True,
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
    )

    if export_file_name != None:
        fig.write_html(
            os.path.join(
                args.work_path,
                "inspect_puncta/{}".format(str(export_file_name) + ".html"),
            )
        )

    fig.show()


# puncta in ROIs
def inspect_puncta_ROI_matplotlib(args, fov, code, position, center_dist=40):
    """Plots identified puncta for a specific fov/code using Matplotlib. Assumes the puncta have already been consolidated accross channels.

    :param args.Args args: configuration options.
    :param int fov: the field of fiew of the volume chunk to be returned.
    :param int code: the code of the volume chunk to be returned.
    :param list position: the center point of the region that should be visualized. Expects position in the format of :math:`[z, y, x]`.
    :param int center_dist: distance from the center that should be viewable. Default: ``40``
    """

    reference = retrieve_all_puncta(args, fov)

    fig, axs = plt.subplots(4, 10, figsize=(15, 7), dpi=100)

    for channel in range(4):
        for z_ind, z in enumerate(np.linspace(position[0] - 10, position[0] + 10, 10)):
            ROI_min = [int(z), position[1] - center_dist, position[2] - center_dist]
            ROI_max = [int(z), position[1] + center_dist, position[2] + center_dist]
            img = retrieve_img(args, fov, code, channel, ROI_min, ROI_max)
            axs[channel, z_ind].imshow(
                img, cmap=plt.get_cmap(args.colorscales[channel]), vmax=150
            )
            if channel == 3:
                axs[channel, z_ind].set_xlabel("{0:0.0f}".format(z))
            if z_ind == 0:
                axs[channel, z_ind].set_ylabel(args.channel_names[channel])

    ROI_min = [position[0] - 10, position[1] - center_dist, position[2] - center_dist]
    ROI_max = [position[0] + 10, position[1] + center_dist, position[2] + center_dist]
    temp = [
        x["code{}".format(code)]
        for x in reference
        if "code{}".format(code) in x
        and in_region(x["code{}".format(code)]["position"], ROI_min, ROI_max)
    ]

    for channel in range(4):
        temp2 = [
            x["c{}".format(channel)]["position"]
            for x in temp
            if "c{}".format(channel) in x
        ]
        for puncta in temp2:
            axs[channel, (puncta[0] - position[0] + 10) // 2].scatter(
                puncta[1] - position[1] + center_dist,
                puncta[2] - position[2] + center_dist,
                marker="o",
                s=30,
            )

    fig.suptitle("Fov {} Code {}".format(fov, code))
    plt.show()


def inspect_puncta_ROI_plotly(
    args,
    fov,
    position,
    c_list=[0, 1, 2, 3],
    center_dist=40,
    spacer=40,
    export_file_name=False,
):
    """Plots identified puncta for a specific fov/code using Plotly. Assumes the puncta have already been consolidated accross channels.

    :param args.Args args: configuration options.
    :param int fov: the field of fiew of the volume chunk to be returned.
    :param list position: the center point of the region that should be visualized. Expects position in the format of :math:`[z, y, x]`.
    :param list c_list: the codes to include in the viaulization.
    :param int center_dist: distance from the center that should be viewable. Default: ``40``
    :param int spacer: scaling factor to use for z-spacing. Default: ``40``
    :param str export_file_name: name of the file to be exported. Default: ``None``
    """
    ROI_min = [position[0] - 10, position[1] - center_dist, position[2] - center_dist]
    ROI_max = [position[0] + 10, position[1] + center_dist, position[2] + center_dist]
    reference = retrieve_all_puncta(args, fov)

    fig = go.Figure()

    for i, code in enumerate(args.codes):

        ## Surface -------------
        for c in c_list:

            for zz in np.linspace(ROI_min[0], ROI_max[0], 7):

                img = retrieve_img(
                    args,
                    fov,
                    code,
                    c,
                    [int(zz), ROI_min[1], ROI_min[2]],
                    [int(zz), ROI_max[1], ROI_max[2]],
                )

                y = list(range(ROI_min[1], ROI_max[1]))
                x = list(range(ROI_min[2], ROI_max[2]))
                z = np.ones((ROI_max[1] - ROI_min[1], ROI_max[2] - ROI_min[2])) * (
                    int(zz) + 0.7 * c + i * spacer
                )
                fig.add_trace(
                    go.Surface(
                        x=x,
                        y=y,
                        z=z,
                        surfacecolor=img,
                        cmin=0,
                        cmax=500,
                        colorscale=args.colorscales[c],
                        showscale=False,
                        opacity=0.2,
                    )
                )

        ## Scatter --------------
        temp = [
            x["code{}".format(code)]
            for x in reference
            if "code{}".format(code) in x
            and in_region(x["code{}".format(code)]["position"], ROI_min, ROI_max)
        ]

        for c in c_list:

            temp2 = [
                x["c{}".format(c)]["position"] for x in temp if "c{}".format(c) in x
            ]
            temp2 = np.asarray(temp2)
            if len(temp2) == 0:
                continue
            fig.add_trace(
                go.Scatter3d(
                    z=temp2[:, 0] + i * spacer,
                    y=temp2[:, 1],
                    x=temp2[:, 2],
                    mode="markers",
                    marker=dict(
                        color=args.colors[c],
                        size=4,
                    ),
                )
            )

    # ------------
    fig.add_trace(
        go.Scatter3d(
            z=[ROI_min[0], ROI_max[0] + (len(args.codes) - 1) * spacer],
            y=[(ROI_min[1] + ROI_max[1]) / 2] * 2,
            x=[(ROI_min[2] + ROI_max[2]) / 2] * 2,
            mode="lines",
            line=dict(
                color="black",
                width=10,
            ),
        )
    )

    # ---------------------
    fig.update_layout(
        title="Inspect fov{}, code: ".format(fov)
        + " ".join([str(x) for x in args.codes]),
        width=800,
        height=800,
        scene=dict(
            aspectmode="data",
            xaxis_visible=True,
            yaxis_visible=True,
            zaxis_visible=True,
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
    )

    if export_file_name != None:
        fig.write_html(
            os.path.join(
                args.work_path,
                "inspect_puncta/{}".format(str(export_file_name) + ".html"),
            )
        )

    fig.show()


# Individual puncta
def inspect_puncta_individual_matplotlib(args, fov, puncta_index, center_dist=40):
    """Plots specified puncta using Matplotlib. Assumes the puncta have already been consolidated accross channels.

    :param args.Args args: configuration options.
    :param int fov: the field of fiew of the volume chunk to be returned.
    :param int puncta_index: the index of the puncta to visualize.
    :param int center_dist: distance from the center that should be viewable. Default: ``40``
    """

    import matplotlib.pyplot as plt

    puncta = retrieve_one_puncta(args, fov, puncta_index)

    fig, axs = plt.subplots(4, len(args.codes), figsize=(15, 11))

    for code_ind, code in enumerate(args.codes):

        if "code{}".format(code) not in puncta:
            for c in range(4):
                fig.delaxes(axs[c, code_ind])
            continue

        position = puncta["code{}".format(code)]["position"]
        ROI_min = [
            int(position[0]),
            position[1] - center_dist,
            position[2] - center_dist,
        ]
        ROI_max = [
            int(position[0]),
            position[1] + center_dist,
            position[2] + center_dist,
        ]
        for c in range(4):
            img = retrieve_img(args, fov, code, c, ROI_min, ROI_max)
            axs[c, code_ind].imshow(
                img, cmap=plt.get_cmap(args.colorscales[c]), vmax=150
            )
            axs[c, code_ind].set_title("code{}".format(code))
            axs[c, code_ind].set_xlabel(
                "{0:0.2f}".format(img[center_dist, center_dist])
            )
            axs[c, code_ind].set_ylabel(args.channel_names[c])

        axs[puncta["code{}".format(code)]["color"], code_ind].scatter(
            center_dist, center_dist, c="white"
        )

    fig.suptitle("fov{} puncta{}".format(fov, puncta_index))
    fig.tight_layout()
    plt.show()


def inspect_puncta_individual_plotly(args, fov, puncta, center_dist=40, spacer=40, save=False):
    r"""Visualizes specified puncta in a 3D space using Plotly. Assumes the puncta have already been consolidated across channels.

    :param args.Args args: configuration options. 
    :param int fov: Identifier of the specific region in the image dataset, the field of view of the volume chunk to be returned.
    :param dict puncta: The puncta to visualize. Should contain information about the puncta's position in the 3D space and its index.
    :param int center_dist: Distance from the center of the puncta that should be viewable in the plot. Default: ``40``
    :param int spacer: Scaling factor used for z-spacing to separate different rounds of imaging in the 3D plot. Default: ``40``
    :param bool save: If True, the plot will be saved as an HTML file in the directory specified by args.work_path. Default: ``False``
    
    This function generates an interactive 3D scatter plot using Plotly, where each puncta is represented as a point in the 3D space. 
    The plot also includes slices of images from different rounds of imaging, providing a contextual understanding of puncta positioning.

    Note:
    The function assumes that the puncta have already been consolidated across different channels and rounds of imaging.
    """

    # Information about the puncta
    reference = retrieve_all_puncta(args,fov)
    # pprint.pprint(puncta)

    # Definition of ROI
    d0, d1, d2 = puncta['position']
    ROI_min = [d0-10, d1-center_dist, d2-center_dist]
    ROI_max = [d0+10, d1+center_dist, d2+center_dist]
   
    N = 5
    fig = go.Figure()
    for i, code in enumerate(args.codes):
 
        # Scatter all puncta -----------------
        puncta_lists = [puncta['code{}'.format(code)] for puncta in reference if 'code{}'.format(code) in puncta and in_region(puncta['code{}'.format(code)]['position'], ROI_min,ROI_max) ] 
        if not puncta_lists:
            continue
        
        for channel in range(4):
            position_list = np.asarray([puncta['c{}'.format(channel)]['position'] for puncta in puncta_lists if 'c{}'.format(channel) in puncta])
            text_list = [puncta['index'] for puncta in puncta_lists if 'c{}'.format(channel) in puncta]
            if len(position_list) == 0:
                continue
            
            # Plot all puncta
            fig.add_trace(
                go.Scatter3d(
                    z = position_list[:,0] + i * spacer,
                    y = position_list[:,1],
                    x = position_list[:,2],
                    text = text_list,
                    mode = 'markers',
                    marker = dict(
                        color = args.colors[channel],
                        size = 4,
                    ),
                    hoverinfo = 'text'
                ) 
            )

            # Visualize the image -------------
            for zz in np.linspace(ROI_min[0], ROI_max[0], N):

                # Retrive image
                ROI_min_temp,ROI_max_temp = ROI_min[:],ROI_max[:]
                ROI_min_temp[0] = zz
                ROI_max_temp[0] = zz
                im = retrieve_img(args,fov,code,channel,ROI_min,ROI_max)
                
                # Set up the image
                y = list(range(ROI_min[1], ROI_max[1]))
                x = list(range(ROI_min[2], ROI_max[2]))
                z = np.ones(
                        (ROI_max[1] - ROI_min[1], ROI_max[2] - ROI_min[2])
                    ) * (int(zz) + 0.5 * channel + i * spacer)
                fig.add_trace(
                    go.Surface(
                        x=x,
                        y=y,
                        z=z,
                        surfacecolor=im,
                        cmin=0,
                        cmax=500,
                        colorscale=args.colorscales[channel],
                        showscale=False,
                        opacity=0.2,
                    )
                )
            
            # Plot this puncta
            if 'code{}'.format(code) not in puncta:
                continue
            if "c{}".format(channel) not in puncta['code{}'.format(code)]:
                continue
            fig.add_trace(
                    go.Scatter3d(
                        z=[puncta['code{}'.format(code)]["c{}".format(channel)]["position"][0] + i * spacer],
                        y=[puncta['code{}'.format(code)]["c{}".format(channel)]["position"][1]],
                        x=[puncta['code{}'.format(code)]["c{}".format(channel)]["position"][2]],
                        text = puncta['index'],
                        mode = "markers",
                        marker=dict(color="gray", size=8, symbol="circle-open"),
                        hoverinfo = 'text'
                    )
                )
    
    nearest_puncta_list = puncta_all_nearest_points(args, puncta)
    # pprint.pprint(nearest_puncta_list)


    for code in range(7):

        if 'code{}'.format(code) not in nearest_puncta_list:
            continue
        nearest_puncta = nearest_puncta_list['code{}'.format(code)]

        for c in range(4):
            if 'c{}'.format(c) not in nearest_puncta:
                continue
            local_maximum = nearest_puncta['c{}'.format(c)]
            z,y,x = local_maximum['position']
            fig.add_trace(
                    go.Scatter3d(
                        z=[z + code * spacer],
                        y=[y],
                        x=[x],
                        text = 'intensity {0:0.2f} Distance {1:0.2f}'.format(local_maximum['intensity'],local_maximum['distance']),
                        mode = "markers",
                        marker = dict(color= args.colors[c], size=12, symbol="square-open"),
                        hoverinfo = 'text'
                    )
                )
  
    
    # Global visualization
    camera = dict(
        eye = dict( x=2, y=2, z=2 )
    )
    fig.update_layout(
        title = "Puncta Fov{} index {} {}".format(fov, puncta['index'], puncta['barcode']),
        width = 800,
        height = 800,
        scene_camera = camera,
        scene = dict(
            aspectmode = 'data',
            xaxis_visible = True,
            yaxis_visible = True, 
            zaxis_visible = True, 
            xaxis_title = "X",
            yaxis_title = "Y",
            zaxis_title = "Z" ,
        )
    )
   
    if save:
        fig.write_html(
            os.path.join(
                args.work_path,
                'inspect_puncta_individual_plotly_fov_{}_puncta_{}.html'.format(fov, puncta['index']))
            )
        
    fig.show()


# Puncta across rounds
def inspect_between_rounds_plotly(
    args, fov, code1, code2, ROI_min, ROI_max, spacer=40, export_file_name=None
):
    """Plots puncta across rounds (for two specified codes) using Plotly.

    :param args.Args args: configuration options.
    :param int fov: the field of fiew of the volume chunk to be returned.
    :param str code1: name of the first code to include in comparison.
    :param str code2: name of the first code to include in comparison.
    :param list ROI_min: minimum coordinates of the volume chunk to display. Expects coordinates in the format of :math:`[z, y, x]`.
    :param list ROI_max: maximum coordinates of the volume chunk to display. Expects coordinates in the format of :math:`[z, y, x]`.
    :param int spacer: scaling factor to use for z-spacing. Default: ``40``
    :param str export_file_name: name of the file to be exported. Default: ``None``
    """

    if ROI_max[0] - ROI_min[0] > 20:
        print("ROI_max[0]-ROI_min[0]should be smaller than 20")
        return

    code1, code2 = "code{}".format(code1), "code{}".format(code2)

    reference = retrieve_all_puncta(args, fov)
    reference = [x for x in reference if in_region(x["position"], ROI_min, ROI_max)]
    print("Only {} puncta remained".format(len(reference)))

    fig = go.Figure()

    # Lines  between codes ====================
    temp = [x for x in reference if (code1 in x) and (code2 in x)]
    for x in temp:
        center1, center2 = x[code1]["position"], x[code2]["position"]
        name = x["index"]
        fig.add_trace(
            go.Scatter3d(
                z=[center1[0], center2[0] + spacer],
                y=[center1[1], center2[1]],
                x=[center1[2], center2[2]],
                mode="lines",
                name=name,
                line=dict(
                    color="gray",
                ),
            )
        )

    # Code1  =========================
    temp = [x for x in reference if (code1 in x)]

    # Centers
    points = [x[code1]["position"] for x in temp]
    points = np.asarray(points)
    texts = ["{} {}".format(x["index"], code1) for x in temp]
    if len(points) > 0:
        fig.add_trace(
            go.Scatter3d(
                z=points[:, 0],
                y=points[:, 1],
                x=points[:, 2],
                text=texts,
                mode="markers+text",
                name="consensus",
                marker=dict(
                    color="gray",
                    size=10,
                    opacity=0.2,
                ),
            )
        )

    # Scatters --------------
    for c in range(4):

        points = [
            x[code1]["c{}".format(c)]["position"]
            for x in temp
            if "c{}".format(c) in x[code1]
        ]
        points = np.asarray(points)
        if len(points) == 0:
            continue

        fig.add_trace(
            go.Scatter3d(
                z=points[:, 0],
                y=points[:, 1],
                x=points[:, 2],
                name="channels",
                mode="markers",
                marker=dict(
                    color=args.colors[c],
                    size=4,
                ),
            )
        )

    # Lines --------------
    for x in temp:
        points = [
            x[code1][c]["position"] for c in ["c0", "c1", "c2", "c3"] if c in x[code1]
        ]

        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):

                fig.add_trace(
                    go.Scatter3d(
                        z=[points[i][0], points[j][0]],
                        y=[points[i][1], points[j][1]],
                        x=[points[i][2], points[j][2]],
                        mode="lines",
                        name="inter channel",
                        line=dict(
                            color="gray",
                        ),
                    )
                )

    # Code2  =========================
    temp = [x for x in reference if (code2 in x)]

    # Centers
    points = [x[code2]["position"] for x in temp]
    points = np.asarray(points)
    texts = ["{} {}".format(x["index"], code2) for x in temp]

    if len(points) > 0:
        fig.add_trace(
            go.Scatter3d(
                z=points[:, 0] + spacer,
                y=points[:, 1],
                x=points[:, 2],
                text=texts,
                mode="markers+text",
                name="consensus",
                marker=dict(
                    color="gray",
                    size=10,
                    opacity=0.2,
                ),
            )
        )

    # Scatters --------------
    for c in range(4):
        points = [
            x[code2]["c{}".format(c)]["position"]
            for x in temp
            if "c{}".format(c) in x[code2]
        ]
        points = np.asarray(points)
        if len(points) == 0:
            continue
        fig.add_trace(
            go.Scatter3d(
                z=points[:, 0] + spacer,
                y=points[:, 1],
                x=points[:, 2],
                mode="markers",
                name="channels",
                marker=dict(
                    color=args.colors[c],
                    size=4,
                ),
            )
        )

    ## Lines --------------
    for x in temp:
        points = [
            x[code2][c]["position"] for c in ["c0", "c1", "c2", "c3"] if c in x[code2]
        ]
        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):
                fig.add_trace(
                    go.Scatter3d(
                        z=[points[i][0] + spacer, points[j][0] + spacer],
                        y=[points[i][1], points[j][1]],
                        x=[points[i][2], points[j][2]],
                        mode="lines",
                        name="inter channel",
                        line=dict(
                            color="gray",
                            # size=4,
                        ),
                    )
                )

    # ---------------------
    fig.update_layout(
        title="Puncta Between Rounds: Fov{} - {} & {}".format(fov, code1, code2),
        width=800,
        height=800,
        showlegend=False,
        scene=dict(
            aspectmode="data",
            xaxis_visible=True,
            yaxis_visible=True,
            zaxis_visible=True,
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
    )

    if export_file_name != None:
        fig.write_html(
            os.path.join(
                args.work_path,
                "inspect_puncta/{}".format(str(export_file_name) + ".html"),
            )
        )

    fig.show()


def inspect_across_rounds_plotly(
    args, fov, ROI_min, ROI_max, spacer=20, export_file_name=None
):
    """Plots puncta across rounds (for all codes) using Plotly.

    :param args.Args args: configuration options.
    :param int fov: the field of fiew of the volume chunk to be returned.
    :param list ROI_min: minimum coordinates of the volume chunk to display. Expects coordinates in the format of :math:`[z, y, x]`.
    :param list ROI_max: maximum coordinates of the volume chunk to display. Expects coordinates in the format of :math:`[z, y, x]`.
    :param int spacer: scaling factor to use for z-spacing. Default: ``20``
    :param str export_file_name: name of the file to be exported. Default: ``None``
    """

    reference = retrieve_all_puncta(args, fov)
    reference = [x for x in reference if in_region(x["position"], ROI_min, ROI_max)]

    fig = go.Figure()

    ## Lines ====================
    for puncta in reference:
        codes = sorted([x for x in puncta if x.startswith("code")])
        for i in range(len(codes) - 1):
            code1 = codes[i]
            code2 = codes[i + 1]

            center1, center2 = puncta[code1]["position"], puncta[code2]["position"]
            name = puncta["index"]
            fig.add_trace(
                go.Scatter3d(
                    z=[
                        center1[0] + int(code1[-1]) * spacer,
                        center2[0] + int(code2[-1]) * spacer,
                    ],
                    y=[center1[1], center2[1]],
                    x=[center1[2], center2[2]],
                    mode="lines",
                    name=name,
                    line=dict(
                        color="gray",
                    ),
                )
            )

    ## Code  =========================
    for code in args.codes:

        code_str = "code{}".format(code)

        temp = [x for x in reference if (code_str in x)]

        ## Centers
        points = [x[code_str]["position"] for x in temp]
        points = np.asarray(points)
        texts = ["{} {}".format(x["index"], code_str) for x in temp]
        if len(points) > 0:
            fig.add_trace(
                go.Scatter3d(
                    z=points[:, 0] + code * spacer,
                    y=points[:, 1],
                    x=points[:, 2],
                    text=texts,
                    mode="markers+text",
                    name="consensus",
                    marker=dict(
                        color="gray",
                        size=10,
                        opacity=0.2,
                    ),
                )
            )

        ## Scatters --------------
        for c in range(4):
            points = [
                x[code_str]["c{}".format(c)]["position"]
                for x in temp
                if "c{}".format(c) in x[code_str]
            ]
            points = np.asarray(points)
            if len(points) == 0:
                continue

            fig.add_trace(
                go.Scatter3d(
                    z=points[:, 0] + code * spacer,
                    y=points[:, 1],
                    x=points[:, 2],
                    name="channels",
                    mode="markers",
                    marker=dict(
                        color=args.colors[c],
                        size=4,
                    ),
                )
            )

        ## Lines --------------
        for x in temp:
            points = [
                x[code_str][c]["position"]
                for c in ["c0", "c1", "c2", "c3"]
                if c in x[code_str]
            ]

            for i in range(len(points) - 1):
                for j in range(i + 1, len(points)):

                    fig.add_trace(
                        go.Scatter3d(
                            z=[
                                points[i][0] + code * spacer,
                                points[j][0] + code * spacer,
                            ],
                            y=[points[i][1], points[j][1]],
                            x=[points[i][2], points[j][2]],
                            mode="lines",
                            name="inter channel",
                            line=dict(
                                color="gray",
                                # size=4,
                            ),
                        )
                    )

    # ---------------------
    fig.update_layout(
        title="Puncta Across Rounds: FOV{}".format(fov),
        width=800,
        height=800,
        showlegend=False,
        scene=dict(
            aspectmode="data",
            xaxis_visible=True,
            yaxis_visible=True,
            zaxis_visible=True,
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
    )

    if export_file_name != None:
        fig.write_html(
            os.path.join(
                args.work_path,
                "inspect_puncta/{}".format(str(export_file_name) + ".html"),
            )
        )

    fig.show()


# TODO fix labels, add legend and handle multi-missing codes. 
def inspect_puncta_improvement_matplotlib(args, fov, puncta_index, option = 'final', center_dist=40, save = False,missing_code=0):
    r"""
    Visualizes puncta improvement using Matplotlib. The function generates a detailed plot of the region 
    of interest (ROI) around a given puncta and shows changes in the puncta position over different 
    rounds of image acquisition. The function supports visualization of missing code if provided.

    :param args: Configuration options, including methods for retrieving puncta and images.
    :type args: args.Args instance 
    :param int fov: The field of view (fov) to consider.
    :param int puncta_index: The index of the puncta to start the search from.
    :param str option: Option for puncta visualization, should be either 'initial', 'intermediate', or 'final'. Default is 'final'.
    :param int center_dist: The distance to the center of the ROI. Default is 40.
    :param bool save: Whether to save the generated plot or not. If set to False, the plot will be displayed. Default is False.
    :param int missing_code: The missing code to consider in the visualization. Default is 0.

    :return: None
    """

    from exm.puncta.improve import puncta_nearest_points

    # Get the FOV all puncta information 
    reference = retrieve_all_puncta(args,fov)

    # Get individual puncta information based on puncta_indexn
    puncta = retrieve_one_puncta(args,fov, puncta_index)
    print('fov',fov,'index',puncta_index)

    # Get postion of the puncta
    d0, d1, d2 = puncta['position']
    print('puncta position',d0,d1,d2)

    # Define the Region of Interest (ROI) based on the puncta position
    ROI_min = [max(0,d0 - 10),max(0,d1 - center_dist), max(d2 - center_dist,0)]
    ROI_max = [d0 + 10,d1 + center_dist, d2 + center_dist]    
    print('ROI_min,ROI_max = {},{}'.format(ROI_min,ROI_max))

    # Define the z-stack step size
    delta_z = (ROI_max[0] - ROI_min[0])/10

    ## Clean old plots
    plt.close()
    
    # If missing code is present, find its new and closest positions
    if missing_code > 0:
        arr = np.array(list(puncta['barcode']))
        missed_code = np.where(arr == '_')[0]
        ref_code, new_position, closest_position = puncta_nearest_points(args,puncta['fov'],puncta['index'],missed_code[0])  
        if new_position:
            print('new_position', new_position)
        if closest_position:
            print('closest_position', closest_position)

    fontsize = 40

    # Initialize Matplotlib figure and subplot grids
    import matplotlib.gridspec as gridspec
    plt.figure(figsize=(20, 45), dpi=100)
    outer = gridspec.GridSpec(7, 1, height_ratios = [1]*7, hspace = .05)

    # For each of the 7 rounds
    for code in range(7):
        # Initialize inner grid for each code
        inner = gridspec.GridSpecFromSubplotSpec(4, 10, subplot_spec = outer[code], hspace = 0)
        
        # For each of the four channel 
        for channel in range(4):

            # For each z-slice in the range of 10
            for z_ind,z in enumerate(np.linspace(ROI_min[0],ROI_max[0],10)):

                ax = plt.subplot(inner[channel,z_ind])
                ax.set_xticks([])
                ax.set_yticks([])
                
                temp_ROI_min, temp_ROI_max = ROI_min[:], ROI_max[:]
                temp_ROI_min[0] = int(z)
                temp_ROI_max[0] = int(z)

                img = retrieve_img(args,fov,code,channel,temp_ROI_min,temp_ROI_max)

                ax.imshow(img, cmap=plt.get_cmap(args.colorscales[channel]),vmin = 0, vmax = 200)
                
                if code == 7 and channel == 3:
                    ax.set_xlabel('{0:0.0f}'.format(z))
                
                # display y-labels when channel is 2 (code{},Search Code,Ref Code) 
                if z_ind == 0:
                    ax.set_ylabel(args.channel_names[channel])
                    if channel == 1:
                        ax.text(-50,20,'code{}'.format(code),fontsize = fontsize)
                    if missing_code >0 and channel == 2 and code == missed_code[0] :
                        if not new_position:
                            ax.text(-50,20,'Search code None',fontsize = fontsize)
                        else:
                            ax.text(-50,20,'Search Code',fontsize = fontsize)
                    if missing_code >0 and channel == 2 and code == ref_code:
                        if not closest_position:
                            ax.text(-50,20,'Ref code None',fontsize = fontsize)
                        else:
                            ax.text(-50,20,'Ref Code',fontsize = fontsize)

            # Closest postions puncta
            if missing_code >0 and closest_position and code == closest_position['code']:

                if channel == closest_position['color']:     
                    temp = closest_position['position']
                    ax = plt.subplot(inner[channel, int(np.floor((temp[0]-ROI_min[0])/delta_z))])
                    ax.text(0,20,temp[2]-ROI_min[2],temp[1]-ROI_min[1],'closest',fontsize = 20)
                    ax.scatter( temp[2]-ROI_min[2],temp[1]-ROI_min[1], marker = 'D', facecolors='none', edgecolors='violet', s = 270, linewidths=3)
                
            # New postions puncta           
            elif missing_code >0 and new_position and code == new_position['code']:
                if channel == new_position['color']:  
                    temp = new_position['position']
                    ax = plt.subplot(inner[channel, int(np.floor((temp[0]-ROI_min[0])/delta_z))]) 
                    ax.scatter( temp[2]-ROI_min[2],temp[1]-ROI_min[1], marker = 'D', facecolors='none', edgecolors='violet', s = 270, linewidths=3)
                    ax.text(temp[2]-ROI_min[2],temp[1]-ROI_min[1],'new',fontsize = 20)

            # Ref code Puncta
            if missing_code >0 and code == ref_code and channel == puncta['code{}'.format(code)]['color']:
                temp = puncta['code{}'.format(code)]['position']
                ax = plt.subplot(inner[channel, int(np.floor((temp[0]-ROI_min[0])/delta_z))])
                ax.scatter( temp[2]-ROI_min[2],temp[1]-ROI_min[1], marker = 'D', facecolors='none', edgecolors='violet', s = 270, linewidths=3)
                ax.text(temp[2]-ROI_min[2],temp[1]-ROI_min[1],'original',fontsize = 20)

        
        ## Show puncta
        if option == 'final':
            filter1 = [x['code{}'.format(code)] for x in reference if 'code{}'.format(code) in x and in_region( x['code{}'.format(code)]['position'], ROI_min,ROI_max) ] 
            for channel in range(4):
                filter2 = [x['c{}'.format(channel)]['position'] for x in filter1 if 'c{}'.format(channel) in x and in_region(x['c{}'.format(channel)]['position'],ROI_min,ROI_max)]
                for temp in filter2:
                    ax = plt.subplot(inner[channel, int(np.floor((temp[0]-ROI_min[0])/delta_z))])
                    ax.scatter(temp[2]-ROI_min[2],temp[1]-ROI_min[1], marker = 'o', facecolors='none', edgecolors='white', s = 180, linewidths=3)

            if 'code{}'.format(code) in puncta:
                for channel in range(4):
                    if 'c{}'.format(channel) in puncta['code{}'.format(code)]:
                        temp = puncta['code{}'.format(code)]['c{}'.format(channel)]['position']
                        if int(np.floor((temp[0]-ROI_min[0])/delta_z))>=10:
                            continue
                        ax = plt.subplot(inner[channel, int(np.floor((temp[0]-ROI_min[0])/delta_z))])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if puncta['code{}'.format(code)]['color'] == channel:
                            # chosen puncta in that round
                            ax.scatter( temp[2]-ROI_min[2],temp[1]-ROI_min[1], marker = 'D', facecolors='none', edgecolors='cyan', s = 270, linewidths=3)
                        else:
                            # other puncta in that round
                            ax.scatter( temp[2]-ROI_min[2],temp[1]-ROI_min[1], marker = 'D', facecolors='none', edgecolors='yellow', s = 270, linewidths=3)

    if save:
        plt.savefig(os.path.join(args.work_path,'inspect_puncta/puncta_improvement_fov{}_puncta{}.jpg'.format(fov,puncta_index)))
        plt.close()
    else:
        plt.show()



def inspect_improved_puncta_plotly(args, fov, puncta,center_dist=40,spacer=40):

    # Information about the puncta
    with open(args.work_path + '/fov{}/improved_puncta_results.pickle'.format(fov),'rb') as f:
        reference = pickle.load(f)

    # Definition of ROI
    d0, d1, d2 = puncta['position']
    ROI_min = [d0-10, d1-center_dist, d2-center_dist]
    ROI_max = [d0+10, d1+center_dist, d2+center_dist]
   
    N = 5
    fig = go.Figure()
    for i, code in enumerate(args.codes):

        # Plot all puncta ---------------
        puncta_lists = [puncta for puncta in reference if 'code{}'.format(code) in puncta and in_region(puncta['code{}'.format(code)]['position'], ROI_min,ROI_max)] 
        if not puncta_lists:
            continue
        for channel in range(4):
            position_list = np.asarray([puncta['code{}'.format(code)]['c{}'.format(channel)]['position'] for puncta in puncta_lists if 'c{}'.format(channel) in puncta])
            text_list = [puncta['index'] for puncta in puncta_lists if 'c{}'.format(channel) in puncta['code{}'.format(code)]]
            if len(position_list) == 0:
                continue
            
            fig.add_trace(
                go.Scatter3d(
                    z = position_list[:,0] + i * spacer,
                    y = position_list[:,1],
                    x = position_list[:,2],
                    text = text_list,
                    mode = 'markers',
                    marker = dict(
                        color = args.colors[channel],
                        size = 4,
                    ),
                    hoverinfo = 'text'
                ) 
            )


        # Visualize the image -------------
        for channel in range(4):

            for zz in np.linspace(ROI_min[0], ROI_max[0], N):

                # Retrive image
                ROI_min_temp,ROI_max_temp = ROI_min[:],ROI_max[:]
                ROI_min_temp[0] = zz
                ROI_max_temp[0] = zz
                im = retrieve_img(args,fov,code,channel,ROI_min,ROI_max)
                
                # Set up the image
                y = list(range(ROI_min[1], ROI_max[1]))
                x = list(range(ROI_min[2], ROI_max[2]))
                z = np.ones(
                        (ROI_max[1] - ROI_min[1], ROI_max[2] - ROI_min[2])
                    ) * (int(zz) + 0.5 * channel + code * spacer)
                
                fig.add_trace(
                    go.Surface(
                        x=x,
                        y=y,
                        z=z,
                        surfacecolor=im,
                        cmin=0,
                        cmax=500,
                        colorscale=args.colorscales[channel],
                        showscale=False,
                        opacity=0.2,
                    )
                )
        

        # Plot this puncta ----------------------------
        if 'code{}'.format(code) not in puncta:
            continue
        
        for channel in range(4):

            if "c{}".format(channel) not in puncta['code{}'.format(code)]:
                continue
            
            if 'ref_code' not in puncta['code{}'.format(code)]:
                fig.add_trace(
                    go.Scatter3d(
                        z=[puncta['code{}'.format(code)]["c{}".format(channel)]["position"][0] + code * spacer],
                        y=[puncta['code{}'.format(code)]["c{}".format(channel)]["position"][1]],
                        x=[puncta['code{}'.format(code)]["c{}".format(channel)]["position"][2]],
                        text = puncta['index'],
                        mode = "markers",
                        marker=dict(color="gray", size=8, symbol="circle-open"),
                        hoverinfo = 'text'
                    )
                )
            elif 'c{}'.format(channel) in puncta['code{}'.format(code)]:
                local_maximum = puncta['code{}'.format(code)]['c{}'.format(channel)]
                z,y,x = local_maximum['position']

                fig.add_trace(
                        go.Scatter3d(
                            z=[z + code * spacer],
                            y=[y],
                            x=[x],
                            text = 'intensity {0:0.2f} Distance {1:0.2f}'.format(local_maximum['intensity'],local_maximum['distance']),
                            mode = "markers",
                            marker = dict(color= args.colors[channel], size=12, symbol="square-open"),
                            hoverinfo = 'text'
                        )
                    )
    
    # Global visualization
    camera = dict(
        eye = dict( x=2, y=2, z=2 )
    )
    fig.update_layout(
        title = "Puncta Fov{} index {} {}".format(fov, puncta['index'], puncta['barcode']),
        width = 800,
        height = 800,
        scene_camera = camera,
        scene = dict(
            aspectmode = 'data',
            xaxis_visible = True,
            yaxis_visible = True, 
            zaxis_visible = True, 
            xaxis_title = "X",
            yaxis_title = "Y",
            zaxis_title = "Z" ,
        )
    )
    # fig.show()
    fig.write_html(os.path.join(args.work_path,'inspect_puncta/inspect_improved_puncta_plotly_fov_{}_puncta_{}.html'.format(fov, puncta['index'])))