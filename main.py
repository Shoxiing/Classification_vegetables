
def print_hi(name):
    print(f'Hi, {name}') 

if __name__ == '__main__':
    print_hi('PyCharm')

from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st




st.title("VEGTABLE CLASSIFICATION")
st.write("")

file_up = st.file_uploader("Upload an image")# type = "jpg")

transform = transforms.Compose(
    [transforms.Resize((64,64)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


model = torch.load('resnet_veg.pt', map_location=torch.device('cpu'))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    model.cuda()


def get_pred(image):

  img_cab = Image.open(image).convert('RGB')

  img_cab_preprocessed = transform(img_cab)
  batch_img_cab_tensor = torch.unsqueeze(img_cab_preprocessed, 0)
  model.eval()

  out = model(batch_img_cab_tensor.to(device))


  with open('classes_veget.txt') as f:
      labels = [line.strip() for line in f.readlines()]

  _, index = torch.max(out, 1)

  percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

  print(labels[index[0]], percentage[index[0]].item())

  _, indices = torch.sort(out, descending=True)
  return ([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("This is ")
    labels = get_pred(file_up)

    for i in labels:
        st.write("Vegetable: ", i[0], ",   Score: ", i[1])
