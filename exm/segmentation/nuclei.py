import os
import tqdm
import h5py
import pickle
import skimage

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from exm.utils import retrieve_img

from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage import measure,morphology
import cv2
# from segment_anything import build_sam, SamAutomaticMaskGenerator


# Segmentation model
def get_predictor():
    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamPredictor,build_sam

    model_type = "vit_h"
    device = "cuda"
    #TODO add model path 
    sam_checkpoint = "sam_vit_h_4b8939.pth"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor


# Nuclei image --> 2D Mask
def predict_z(args, z,fov, predictor, code=0,channel=4):
    
    # Retrieve image of this layer
    ROI_min = [z,0,0]
    ROI_max = [z,2048,2048]
    image = retrieve_img(args,fov,code,channel,ROI_min,ROI_max)

    # Find all local maximum
    blur = gaussian_filter(image, 10, mode='reflect', cval=0)
    coords = peak_local_max(blur, min_distance = 25, threshold_abs=150,exclude_border=False)
    coords = [[x,y] for x,y in coords if np.mean(image[x-3:x+3,y-3:y+3])>150]
    coords = [list(x[::-1]) for x in coords]

    # Prepare the segmentation predictor
    sl = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype('uint8')
    predictor.set_image(sl)

    # For each local maximum, predict the masks at that location
    mask_list = []
    for i in range(len(coords)):

        # For each local maximum, predict the masks at that location
        input_point = np.asarray([coords[i]])
        input_label = np.asarray([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Check the first three masks
        for k in range(2):
            if scores[k]<0.85 or 1000>np.sum(masks[k]) or np.sum(masks[k])>50000:
                continue

            # See if it contains multiple component, if having more than 3, likely is the background
            labels = skimage.measure.label(masks[k])
            if np.max(labels)>3:
                continue

            # Deal with each of the individual component
            for j in range(1, labels.max()+1):

                # Create a mask for this component
                small_mask = (labels == j)
                area = np.sum(small_mask)
                if 1100>area or area>40000 or np.sum(image*small_mask)/area<160:
                    continue

                mask_list.append(small_mask)
                 
    
    if not mask_list:
        mask_list = np.zeros((1,2048,2048))
    else:
        mask_list = np.array(mask_list)

    with open(args.raw_data_path + 'nuclei/mask/mask_fov_{}_z_{}.pickle'.format(fov,z),'wb') as f:
        pickle.dump(mask_list, f)
        
    # Watershed algorithm
    # mask_2d = np.any(mask_list,axis = 0)

    # distance = ndi.distance_transform_edt(mask_2d)
    # new_coords = peak_local_max(distance, min_distance = 20, labels=mask_2d)

    # mask = np.zeros(distance.shape, dtype=bool)
    # mask[tuple(new_coords.T)] = True
    # markers, _ = ndi.label(mask)
    # labels = watershed(-distance, markers, mask=mask_2d)            

    # new_mask_list = []
    # for m in range(1,np.max(labels)+1):

    #     temp = labels == m
    #     area = np.sum(temp)
    #     if 1000>area or area>40000 or np.sum(image*temp)/area<150:
    #         continue
    
    #     temp = skimage.morphology.erosion(temp,footprint=disk(5))
    #     new_mask_list.append(temp)

    # mask_2d = np.any(new_mask_list,axis = 0)

    # fig,axs = plt.subplots(1,2,figsize = (20,20))
    # axs[0].imshow(labels)
    # axs[1].imshow(mask_2d)
    # plt.title(z)
    # plt.show()


# Nuclei volume --> 3D Mask
def generate_3d_mask(args, fov):

    os.makedirs(args.raw_data_path + 'nuclei/', exist_ok=True)   
    os.makedirs(args.raw_data_path + 'nuclei/mask/', exist_ok=True)   
    os.makedirs(args.raw_data_path + 'nuclei/mesh/', exist_ok=True)   
    os.makedirs(args.raw_data_path + 'nuclei/labels/', exist_ok=True)   

    predictor = get_predictor()

    # Get z max
    with h5py.File(args.h5_path.format(0,4), "r") as f:
        z_max = f['405'].shape[0]

    # Generate masks per z
    for z in tqdm.tqdm(range(z_max)):
        predict_z(args, z,fov,predictor)

    # Generate 3d masks
    mask_3d = []
    for z in tqdm.tqdm(range(z_max)):
        with open(args.raw_data_path + 'nuclei/mask/mask_fov_{}_z_{}.pickle'.format(fov,z),'rb') as f:
            mask_2d = pickle.load(f)
        mask_2d = np.any(mask_2d, axis=0)
        mask_3d.append(mask_2d)

    with open(args.raw_data_path + 'nuclei/mask/mask_fov_{}.pickle'.format(fov),'wb') as f:
        pickle.dump(mask_3d, f)


# Voxel grid --> Nuclei
def retrieve_nuclei_per_fov(args,fov,replace = False):
    
    filename = args.raw_data_path + 'nuclei/labels/labels_fov_{}.pickle'.format(fov)
    if os.path.exists(filename) and replace == False:
        with open(filename,'rb') as f:
            labels =  pickle.load(f)
            return labels
    
    # Load dataset
    with open(args.raw_data_path + 'nuclei/mask/mask_fov_{}.pickle'.format(fov),'rb') as f:
        voxel_grid = pickle.load(f)

    for i, arr in enumerate(voxel_grid):
        voxel_grid[i] = arr.astype(bool)

    voxel_grid  = np.asarray(voxel_grid)  
    # Define structural element
    selem = np.asarray([[[1]],[[1]],[[1]],[[1]],[[1]]])

    # Perform binary dilation on the image
    dilated_image = morphology.binary_dilation(voxel_grid, footprint=selem)
    labels = skimage.measure.label(dilated_image)  
    
    with open(filename,'wb') as f:
        pickle.dump(labels, f)

    return labels


# Generate N random colors
def generate_n_colors(n_colors,plot = False):
    
    import random
    colors = plt.cm.rainbow(np.linspace(0, 1, n_colors+1))
    colors = np.random.permutation(colors)

    if plot:
        plt.close()
        plt.figure()
        for i in range(n_colors):
            plt.plot(i,i,'*',color = colors[i] )
        plt.show()
    return colors


# Plot Nuclei + Puncta
def plot_nuclei_puncta(args, fov, modality = 'mesh', option = 'full',valid_genes=False):

    print('Plot Nuclei + Puncta Fov {} Modality {} Option {}'.format(fov,modality,option))
    
    fig = go.Figure()

    # Load puncta ==============================
    if option == 'original':
        with open(args.puncta_path + 'fov{}/puncta_with_gene.pickle'.format(fov), 'rb') as f:
            puncta_list= pickle.load(f)
        if valid_genes:
            puncta_list = [x for x in puncta_list if x['gene'] != 'N/A']

    elif option == 'improve':
        # Select only valid genes vs show all puncta
        with open(args.puncta_path + 'fov{}/improved_puncta_with_gene.pickle'.format(fov), 'rb') as f:
            puncta_list = pickle.load(f)
        if valid_genes:
            puncta_list = [x for x in puncta_list if x['gene'] != 'N/A']


    # ====================================
    # Plot puncta 
    position = [puncta['position'] for puncta in puncta_list]
    position = np.asarray(position)
    text = ['puncta {} barcode {} gene {}'.format(puncta['index'], puncta['barcode'], puncta['gene']) for puncta in puncta_list]
    if puncta_list:
        fig.add_trace(
            go.Scatter3d(
                x = position[:,0],
                y = position[:,1],
                z = position[:,2],
                mode='markers',
                marker=dict(
                    size=2, 
                    color='red',
                    opacity=0.3
                ),
                text = text,
                hoverinfo = 'text'
            )
        )

    # ====================================
    # show nuclei
    if modality == 'mesh':

        with open(args.raw_data_path + 'nuclei/mask/mask_fov_{}.pickle'.format(fov),'rb') as f:
            voxel_grid = pickle.load(f)
        voxel_grid  = np.asarray(voxel_grid)
        
        # Use marching cubes to convert the mask to a mesh
        vertices, faces, _, _ = measure.marching_cubes(voxel_grid, level=0.5,step_size = 3)

        filename = args.raw_data_path + '/nuclei/mesh/vertices_fov_{}.pickle'.format(fov)
        with open(filename,'wb') as f:
            pickle.dump(vertices,f)
        
        filename = args.raw_data_path + '/nuclei/mesh/faces_fov_{}.pickle'.format(fov)
        with open(filename,'wb') as f:
            pickle.dump(faces,f)

        filename = args.raw_data_path + '/nuclei/mesh/vertices_fov_{}.pickle'.format(fov)
        with open(filename,'rb') as f:
            vertices = pickle.load(f)
        
        filename = args.raw_data_path + '/nuclei/mesh/faces_fov_{}.pickle'.format(fov)
        with open(filename,'rb') as f:
            faces = pickle.load(f)


        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightblue',
                opacity = 0.1
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, 450]), 
                yaxis=dict(range=[0, 2048]), 
                zaxis=dict(range=[0, 2048])
            )
        )

        # Save the figure as an HTML file
        from plotly.offline import plot
        plot(fig, filename= os.path.join(args.puncta_path,'fov{}/plotly_nuclei_fov_{}_mesh_{}.html'.format(fov,fov,option)))


    # ====================================
    if modality == 'labels':
        
        labels = retrieve_nuclei_per_fov(args,fov,replace = True)
        # filename = args.raw_data_path + 'nuclei/labels_fov_{}.pickle'.format(fov)
        # with open(filename,'rb') as f:
        #     labels = pickle.load(f)


        position_value_pairs = [[*position, value] for position, value in np.ndenumerate(labels[::4,::10,::10]) if value != 0]
        position_value_pairs = np.asarray(position_value_pairs)

        position_value_pairs[:,0] *= 4
        position_value_pairs[:,1] *= 10
        position_value_pairs[:,2] *= 10

        filename = args.raw_data_path + 'nuclei/position_value_pairs_{}.pickle'.format(fov)
        with open(filename,'wb') as f:
            pickle.dump(position_value_pairs,f)

        filename = args.raw_data_path + 'nuclei/position_value_pairs_{}.pickle'.format(fov)
        with open(filename,'rb') as f:
            position_value_pairs = pickle.load(f)

        colors = generate_n_colors(np.max(position_value_pairs[:,3]))

        fig.add_trace(
            go.Scatter3d(
                x = position_value_pairs[:,0],
                y = position_value_pairs[:,1],
                z = position_value_pairs[:,2],
                mode='markers',
                marker_color = [colors[x] for x in position_value_pairs[:,3]],
                marker = dict(
                    size = 2,
                    opacity = 0.1
                ),
                text = ['nuclei {}'.format(x) for x in position_value_pairs[:,3]],
                hoverinfo = 'text'
            ),
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, 450]), 
                yaxis=dict(range=[0, 2048]), 
                zaxis=dict(range=[0, 2048])
            )
        )

        # Save the figure as an HTML file
        from plotly.offline import plot
        print('figures/plotly_nuclei_fov_{}_labels_{}.html'.format(fov,option))
        plot(fig, filename=os.path.join(args.puncta_path,'fov{}/plotly_nuclei_fov_{}_labels_{}.html'.format(fov,fov,option)))

