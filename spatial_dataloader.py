import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure

class spatial_dataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform=None):
 
        self.keys = dic.keys()
        self.values = dic.values()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_image(self, video_name, index):
        
        if self.mode == 'train':
            path = os.path.join(self.root_dir, "training", video_name)
        elif self.mode == 'test':
            path = os.path.join(self.root_dir, "validation", video_name)
        else:
            raise ValueError('There are only train and val mode')

        img = Image.open(os.path.join(path,  str('%05d'%(index)) + '.jpg'))
        try:
            transformed_img = self.transform(img)
        except:
            print(os.path.join(path,  str('%05d'%(index)) + '.jpg'))
            img.close()

        return transformed_img

    def __getitem__(self, idx):

        if self.mode == 'train':
            #print("self.keys: ", self.keys)
            video_name, nb_clips = list(self.keys)[idx].split(' ')
            nb_clips = int(nb_clips)
            clips = []
        
            clips.append(random.randint(1, int(nb_clips/3)))
            clips.append(random.randint(int(nb_clips/3), int(nb_clips*2/3)))
            clips.append(random.randint(int(nb_clips*2/3), nb_clips+1))
            
        elif self.mode == 'test':
            video_name, index = list(self.keys)[idx].split(' ')
            index =abs(int(index))
        else:
            raise ValueError('There are only train and val mode')

        label = list(self.values)[idx]
        #label = int(label)-1
        
        if self.mode=='train':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data[key] = self.load_image(video_name, index)
                    
            sample = (data, label)
        elif self.mode=='test':
            data = self.load_image(video_name, index)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path, train_list, test_list):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.train_frame_count ={}
        self.test_frame_count = {}
        self.train_video = {}
        self.test_video = {}
        self.train_list = train_list
        self.test_list = test_list

    def load_frame_count(self, split):
        #print '==> Loading frame number of each video'
        #with open('dic/frame_count.pickle','rb') as file:
        #    dic_frame = pickle.load(file)
        #file.close()
        if split == 'train':
        	split_list = self.train_list
        else:
        	split_list = self.test_list
        lines = [line.strip() for line in open(split_list).readlines()]

        for line in lines:
            videoname = line.split(' ')[0]
            label = int(line.split(' ')[1])
            num_imgs = int(line.split(' ')[2])
            if split == 'train':
                video_path = os.path.join(self.data_path, "training", videoname)
                self.train_frame_count[videoname] = num_imgs
                self.train_video[videoname] = label
            else:
                video_path = os.path.join(self.data_path, "validation", videoname)
                self.test_frame_count[videoname] = num_imgs
                self.test_video[videoname] = label


    def run(self):
        self.load_frame_count('train')
        self.load_frame_count('test')
        self.get_training_dic()
        self.val_sample()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def get_training_dic(self):
        #print '==> Generate frame numbers of each training video'
        self.dic_training={}
        for video in self.train_video:
            #print videoname
            nb_frame = self.train_frame_count[video]-10+1
            key = video+' '+ str(nb_frame)
            self.dic_training[key] = self.train_video[video]
                    
    def val_sample(self):
        print('==> sampling testing frames')
        self.dic_testing={}
        data_dir = "/media/lili/fce9875a-a5c8-4c35-8f60-db60be29ea5d/Moments_in_Time_Raw/validation"
        for video_name in self.test_video:
            # frame_dir_paths = os.path.join(data_dir, video_name)
            # frame_paths = []
            # for frame in os.listdir(frame_dir_paths):
            #     if 'jpg' in frame:
            #         frame_paths.append(os.path.join(frame_dir_paths, frame))
            # frames = [Image.open(frame).convert('RGB') for frame in frame_paths]
            for i in range(6):
                frame = i+2
                key = video_name+ ' '+str(frame+1)
                self.dic_testing[key] = self.test_video[video_name]      

    def train(self):
        training_set = spatial_dataset(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                transforms.Scale([340,256]),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print('==> Training data :',len(training_set),'frames')
        print(training_set[1][0]['img1'].size())

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = spatial_dataset(dic=self.dic_testing, root_dir=self.data_path, mode='test', transform = transforms.Compose([
                transforms.Scale([256, 256]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print('==> Validation data :',len(validation_set),'frames')
        print(validation_set[1][1].size())

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader


if __name__ == '__main__':
    
   dataloader = spatial_dataloader( BATCH_SIZE=16,
                                    num_workers=8,
                                    path='/media/lili/fce9875a-a5c8-4c35-8f60-db60be29ea5d/Moments_in_Time_Raw/',
                                    train_list ='./img_list/new_moments_train_list.txt',
                                    test_list = './img_list/new_moments_validation_list.txt')
   train_loader,val_loader,test_video = dataloader.run()
