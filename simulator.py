import os
import sys

sys.path.append('./build')

import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import Tkinter

import MatterSim


class MySimulator():

    def __init__(self, root_path, scan):
        self.sim = MatterSim.Simulator()
        self.width = 640
        self.height = 640
        self.vfov = math.radians(90.0)
        self.sim.setCameraResolution(self.width, self.height)
        self.sim.setCameraVFOV(self.vfov)
        self.sim.setElevationLimits(-30 * math.pi / 180.0, 30 * math.pi / 180.0)
        self.sim.setRenderingEnabled(False)
        self.sim.init()

        self.colorImagePathPrefix = os.path.join(root_path, 'sampled_color_images', scan)
        self.depthImagePathPrefix = os.path.join(root_path, 'sampled_depth_images', scan)
        self.semanticImagePathPrefix = os.path.join(root_path, 'sampled_semantic_images', scan)

        self.scan = scan
        self.location = ''
        self.heading = 0.0
        self.elevation = 0.0
        self.viewIndex = 0
        self.step = 0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.topViewMaxPixels = 800
        self.topViewImageScale = 1
        self.topViewHeightScale = 0.6
        self.colorImage = None
        self.depthImage = None
        self.semanticImage = None
        self.topViewColorMap = None
        self.topViewSemanticMap = None
        self.spacePointColor = None
        self.spacePointSemantic = None

    def stateReset(self, location, heading, elevation):
        # Initiate the state of simulator, input the heading and elevation in degree
        self.colorImage = None
        self.depthImage = None
        self.semanticImage = None
        self.topViewColorMap = None
        self.topViewSemanticMap = None
        self.spacePointColor = None
        self.spacePointSemantic = None
        self.location = ''
        self.heading = 0.0
        self.elevation = 0.0
        self.step = 0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0



        heading = round(heading / 30.0) * 30
        elevation = round(elevation / 30.0) * 30
        self.sim.newEpisode(self.scan, location, math.radians(heading), math.radians(elevation))
        self.updateState()

    def makeAction(self, action):
        location = 0
        heading = 0
        elevation = 0

        if action == 'f':
            location = 1
        elif action == 'a':
            heading = math.radians(-30)
        elif action == 'd':
            heading = math.radians(30)
        elif action == 'w':
            elevation = math.radians(30)
        elif action == 's':
            elevation = math.radians(-30)

        self.sim.makeAction(location, heading, elevation)

        self.updateState()

    def updateState(self):
        state = self.sim.getState()
        self.location = state.location.viewpointId
        self.heading = round(state.heading / math.pi * 180.0)
        self.elevation = round(state.elevation / math.pi * 180.0)
        self.step = state.step
        self.x = state.location.point[0]
        self.y = state.location.point[1]
        self.z = state.location.point[2]
        self.viewIndex = ((self.elevation + 30) / 30) * 12 + (self.heading / 30)

        self.colorImagePath = os.path.join(self.colorImagePathPrefix, self.location + '_%d.jpg' % self.viewIndex)
        self.depthImagePath = os.path.join(self.depthImagePathPrefix, self.location + '_%d.png' % self.viewIndex)
        self.semanticImagePath = os.path.join(self.semanticImagePathPrefix, self.location + '_%d.png' % self.viewIndex)

        self.colorImage = plt.imread(self.colorImagePath)
        self.depthImage = plt.imread(self.depthImagePath)
        self.semanticImage = (plt.imread(self.semanticImagePath) * 255).astype('uint8')

        self.topViewColorMap = self.calculateTopView('Color', self.colorImage, self.depthImage)
        self.topViewSemanticMap = self.calculateTopView('Semantic', self.semanticImage, self.depthImage)

        center = (self.width // 2, self.height // 2)
        M = cv2.getRotationMatrix2D(center, self.heading, 1.0)
        self.topViewColorMap = cv2.warpAffine(self.topViewColorMap, M, (self.width, self.height))
        self.topViewSemanticMap = cv2.warpAffine(self.topViewSemanticMap, M, (self.width, self.height))

        # self.displayCurrentView()


    def displayCurrentView(self):
        plt.figure(figsize=(10,10))
        plt.subplot(2, 3, 1)
        plt.imshow(self.colorImage)
        plt.subplot(2, 3, 2)
        plt.imshow(self.depthImage)
        plt.subplot(2, 3, 3)
        plt.imshow(self.semanticImage)
        plt.subplot(2, 3, 5)
        plt.imshow(self.topViewColorMap)
        plt.subplot(2, 3, 6)
        plt.imshow(self.topViewSemanticMap)
        plt.ioff()
        # plt.show()
        plt.savefig('./display.png')

    def rotate_matrix(self,ang):
        # ang = float(-ang)
        # rad = (ang / 180) * np.pi
        rad = -ang
        mat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

        return mat

    def extrinsic_matrix(self, coor, heading, elevation, Tx, Ty, Tz):
        heading = heading / 180.0 * np.pi  # convert degree to rad
        elevation = elevation / 180.0 * np.pi

        i = np.array([1, 0, 0]).reshape(3, 1).astype('float32')
        j = np.array([0, 0, 1]).reshape(3, 1).astype('float32')
        k = np.array([0, -1, 0]).reshape(3, 1).astype('float32')

        I = np.array([1, 0, 0]).reshape(3, 1).astype('float32')
        J = np.array([0, 1, 0]).reshape(3, 1).astype('float32')
        K = np.array([0, 0, 1]).reshape(3, 1).astype('float32')

        M = self.rotate_matrix(heading)
        i[0:2,0] = M.dot(i[0:2,0])
        k[0:2,0] = M.dot(k[0:2,0])

        if elevation != 0:
            dk = np.tan(-1.0 * elevation)
            dj = -1.0 * k * np.tan(-1.0 * elevation)
            j += dj
            k[2,0] += dk

            j /= np.linalg.norm(j)
            k /= np.linalg.norm(k)


        IJK = np.concatenate((np.transpose(I),np.transpose(J),np.transpose(K)),axis=0)
        ijk = np.concatenate((i,j,k),axis=1)
        extr = IJK.dot(ijk)
        T = np.array([Tx,Ty,Tz]).reshape(3,1).astype('float32')
        extr = np.concatenate((extr,T),axis=1)
        extr = np.concatenate((extr,np.zeros((1,4))),axis=0)
        coor = np.concatenate((coor,np.ones((coor.shape[0],1))),axis=1)
        coor = extr.dot(np.transpose(coor))
        coor = np.transpose(coor[0:3,:])


        return coor

    def calculateTopView(self, type, colorImage, depthImage):

        max_pixels = self.topViewMaxPixels
        img_scale = self.topViewImageScale
        height_scale = self.topViewHeightScale

        colorImage = cv2.resize(colorImage, (0, 0), fx=img_scale, fy=img_scale)
        depthImage = cv2.resize(depthImage, (0, 0), fx=img_scale, fy=img_scale)
        (H, W, cc) = colorImage.shape

        # cut off depth
        mask = (depthImage >= 3.0 * depthImage.mean())
        depthImage[mask] = 0.0


        # Calculate 3D coordinates
        CAMERA_FACTOR = 4000.0

        cx = 320.0 * img_scale
        cy = 320.0 * img_scale
        fx = 320.0 * img_scale
        fy = 320.0 * img_scale

        pz = depthImage * 65535 / CAMERA_FACTOR  # pz of size  H * W * 1
        mesh_px = np.repeat(np.array(range(0, W)).reshape(1, -1), H, axis=0).astype('float32')
        px = (mesh_px - cx) * pz / fx

        mesh_py = range(0, H)
        mesh_py.reverse()
        mesh_py = np.repeat(np.array(mesh_py).reshape(-1, 1), W, axis=1).astype('float32')
        py = (mesh_py - cy) * pz / fy
        pz = -1 * pz

        px = px.reshape(H, W, 1)
        py = py.reshape(H, W, 1)
        pz = pz.reshape(H, W, 1)

        # px, py, pz = px, -pz, py

        coor = np.concatenate((px, py, pz), axis=2).astype('float32')
        space_points = coor.reshape(-1, 3)
        color = colorImage.reshape(-1, 3)

        # convert coordinates from egocentric to allocentric
        space_points = self.extrinsic_matrix(space_points,heading=self.heading,elevation=self.elevation, Tx=self.x, Ty=self.y, Tz=self.z)

        # integrade into previous space points
        if type == 'Color':

            if self.spacePointColor is None:
                self.spacePointColor = {'coor': space_points.copy(),
                                        'color': color.copy()}
            else:
                space_points = np.concatenate((self.spacePointColor['coor'], space_points), axis=0)
                color = np.concatenate((self.spacePointColor['color'], color), axis=0).copy()
                self.spacePointColor = {'coor': space_points.copy(),
                                        'color': color.copy()}
        elif type == 'Semantic':

            if self.spacePointSemantic is None:
                self.spacePointSemantic = {'coor': space_points.copy(),
                                           'color': color.copy()}
            else:
                space_points = np.concatenate((self.spacePointSemantic['coor'], space_points), axis=0)
                color = np.concatenate((self.spacePointSemantic['color'], color), axis=0).copy()
                self.spacePointSemantic = {'coor': space_points.copy(),
                                           'color': color.copy()}

        # add current point
        # x_range = np.linspace(self.x - 1, self.x + 1, 100).reshape(100,1)
        # y_range = np.linspace(self.y - 1, self.y + 1, 100).reshape(100,1)
        # z_range = np.linspace(self.z - 1, self.z + 1, 100).reshape(100,1)
        # color_ego = np.zeros((100,3)).astype('uint8')
        # color_ego[:,0] = 255
        # ego_position = np.concatenate((x_range,y_range,z_range),axis=1)
        #
        # space_points = np.concatenate((space_points,ego_position),axis=0)
        # color = np.concatenate((color,color_ego),axis=0)
        #


        x_min = np.min(space_points[:, 0])
        y_min = np.min(space_points[:, 1])
        z_min = np.min(space_points[:, 2])

        space_points[:, 0] -= x_min
        space_points[:, 1] -= y_min
        space_points[:, 2] -= z_min

        x_max = np.max(space_points[:, 0])
        y_max = np.max(space_points[:, 1])
        z_max = np.max(space_points[:, 2])

        if x_max > y_max:
            zoom_in = max_pixels / x_max
        else:
            zoom_in = max_pixels / y_max

        space_points[:, 0] *= zoom_in
        space_points[:, 1] *= zoom_in
        space_points[:, 2] *= zoom_in

        space_points = np.ceil(space_points).astype('int32')

        x_size = np.ceil(x_max * zoom_in).astype('int32')
        y_size = np.ceil(y_max * zoom_in).astype('int32')
        z_size = np.ceil(z_max * zoom_in).astype('int32')

        HEIGHT_TRESHOLD = y_size * height_scale
        top_view = np.zeros((y_size + 2, x_size + 2, 3)).astype('uint8')

        x = space_points[:, 0]
        y = space_points[:, 1]
        z = space_points[:, 2]

        mask = (z <= HEIGHT_TRESHOLD).reshape(-1)
        y = y[mask]
        x = x[mask]

        y = y_size - 1 - y
        top_view[y, x, :] = color[mask, :]
        # top_view[z, x, :] = color[:, :]

        return top_view

class GUI():
    def __init__(self, root_path, scan, loc, heading, elevation):
        self.sim = MySimulator(root_path, scan)
        self.sim.stateReset(loc,heading,elevation)

        self.win = Tkinter.Tk()
        # self.frame = Tkinter.Frame(self.win)

        self.view = Tkinter.Label(self.win, width=1500, height=1500)
        # self.view.grid(row=1, column=1)

        im = Tkinter.PhotoImage(file='./display.png',master=self.win)
        self.view.configure(image=im)
        self.view.image = im

        # self.frame.pack()
        self.view.pack()
        self.win.bind('<Key>', self.read_key)
        self.win.mainloop()


    def read_key(self, event):
        action = ''
        if event.char == 'w':
            action = 'w'
        elif event.char == 's':
            action = 's'
        elif event.char == 'a':
            action = 'a'
        elif event.char == 'd':
            action = 'd'
        elif event.char == 'f':
            action = 'f'

        if action != '':
            self.sim.makeAction(action)
            im = Tkinter.PhotoImage(file='./display.png',master=self.win)
            self.view.configure(image=im)
            self.view.image = im




if __name__ == '__main__':
    root_path = '/media/psf/Dongqiyuan/matterport_dataset'
    scan = '1LXtFkjw3qL'
    loc = '14a8edbbe4b14a05b1b5782a884fb6bf'
    # sim = MySimulator(root_path, scan)
    # sim.stateReset(loc, 0, 0)
    gui = GUI(root_path,scan,loc,0,0)
