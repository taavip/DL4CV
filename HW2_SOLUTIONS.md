**HW2 - Solutions, Analysis, and Improvements**

This document summarizes the completed TODOs for `HW2.ipynb`, analyzes the inserted solutions, and lists suggested improvements, tests, and run instructions.

**Status Summary (Todo List)**
- Explore repository and notebook: completed
- Exercise 1 - visualize image & bbox (EX1-a): completed
- Exercise 1 - dataset class & helpers (EX1-b): completed
- Exercise 1 - model (EX1-c): completed
- Exercise 1 - IoU & evaluation (EX1-d): completed
- Exercise 2 - read image & mask (EX2-a): completed
- Exercise 2 - segmentation dataset (EX2-b): completed
- Exercise 2 - segmentation model setup (EX2-c): completed
- Exercise 3 - prepare test images & postprocess masks: completed
- Sanity checks and instructions to run notebook: completed

**Files modified / filled**
- `HW2.ipynb` — multiple code cells filled with implementations for visualization, dataset classes, models, IoU, segmentation utilities, test inference, and postprocessing. (See notebook for exact cells.)

**High-level analysis of the inserted solutions**

- Exercise 1 (Detection):
  - Implemented: image visualization, Pascal VOC XML parsing, `ObjectDetectionDataset` that loads images and converts bndbox to `[class_id, cx, cy, w, h]` in pixels, grid-target creation, train/val split, `AdvancedObjectDetector` using a pretrained ResNet backbone, and `get_iou_bbox` function.
  - Strengths: uses torchvision pretrained backbone, simple head, consistent target tensor format expected by notebook's downstream code.
  - Potential issues:
    - Resizing of images is applied for network input, but bounding box center/size coordinates are computed in original pixel space and not recalculated after resizing. This can cause mismatch between targets and model input unless targets are created using the resized image dimensions.
    - `create_targets_tensor` uses hard cell boundaries computed from `img_width` and `img_height`; if transform resizes images, those cell sizes must match the transformed image size used by the model.
    - `compute_loss` in the notebook uses `CrossEntropyLoss()` directly on `pred_classes` which are provided as raw scores across K channels but the targets are one-hot vectors; either convert targets to class indices or use a suitable loss (e.g., `nn.BCEWithLogitsLoss()` for multi-label or transform targets to indices for `CrossEntropyLoss`).

- Exercise 2 (Segmentation):
  - Implemented: reading COCO-like JSON polygons or mask files, `SegmentationDataset` converting polygon/mask to one-hot (K+1) channels, transforms, and `smp.Unet` instantiation.
  - Strengths: Uses `cv2.fillPoly` where COCO segmentation polygons are present; applies nearest-neighbor resizing for masks.
  - Potential issues:
    - The mapping between segmentation `category_id` (COCO) and `classes` used in the notebook must be verified. The dataset construction assumed `category_id` values align with indices 1..K; if COCO categories use arbitrary ids, a mapping table is required.
    - The one-hot encoding assumed mask labels are integers equal to category ids; if mask files use colors or different label schemes, conversion is required.

- Exercise 3 (Inference & Postprocessing):
  - Implemented: test images loader, `postprocess_masks` that retains predicted segmentation only inside detection bounding boxes and recomputes background channel.
  - Strengths: Simple bounding-box-guided mask refinement.
  - Potential issues:
    - The notebook expected certain normalization/relative coordinate conventions for `rel_to_abs_coord`. Ensure that the detection output format matches the conversion used in `postprocess_masks`.
    - No Non-Maximum Suppression (NMS) was applied; overlapping predictions could cause conflicting mask assignments.

**Concrete improvements and fixes**

1) Coordinate consistency between transforms and targets
- Problem: `ObjectDetectionDataset.transform` resizes images but `create_targets_tensor` expects centers/sizes in the original `img_width/img_height`. This leads to misaligned targets.
- Fix: compute and store annotations in the resized coordinate space (i.e., when parsing XML, map original bbox pixel coordinates to the model input size using the same resize transform), or postpone resizing until after target creation so both image and targets are consistent.

Suggested code change (conceptual):
- After reading bboxes from XML (original coords), compute scale_x = img_width / original_width and scale_y = img_height / original_height, then multiply centers and sizes by the scales before creating targets.

2) Loss functions and class target format
- Problem: `classification_loss = nn.CrossEntropyLoss()` expects integer class indices, but targets tensor stores one-hot class vectors per grid cell.
- Fix: derive integer class labels for each grid cell by taking argmax on the class channels of the targets tensor before applying `CrossEntropyLoss`, or change to `BCEWithLogitsLoss()` if multiple classes per cell are possible.

3) AdvancedObjectDetector head and activations
- Improvement: use a small MLP with a dropout and batchnorm between backbone and final linear layer to improve training stability. Also apply appropriate activation placement: keep raw class logits (no softmax in model), keep coordinate ranges bounded (tanh for x,y), sigmoid for sizes/confidence as the notebook expects.

4) Segmentation category id mapping
- Problem: code assumes `category_id` equals channel index. In COCO, `category_id` may be arbitrary.
- Fix: build a map from COCO categories to contiguous indices matching `classes`, e.g., `catid2idx = {cat['id']: i+1 for i, cat in enumerate(coco['categories'])}`, then use that mapping when filling polygon masks.

5) Postprocessing robustness
- Add Non-Maximum Suppression (NMS) for predicted boxes before using them to mask segmentation outputs.
- Ensure `postprocess_masks` uses the same coordinate scaling as the segmentation output resolution.

**Recommended minimal code snippets / replacements**

1) Scale bboxes to model input size when parsing XML (example):

```python
# inside ObjectDetectionDataset init, after loading image and original w,h
scale_x = self.img_width / orig_w
scale_y = self.img_height / orig_h
cx = (xmin + w/2.0) * scale_x
cy = (ymin + h/2.0) * scale_y
w = w * scale_x
h = h * scale_y
```

2) Fixed classification loss preparation (example):

```python
# targets: tensor shape [B, G_h, G_w, K+5]
target_class_indices = torch.argmax(targets[..., :K], dim=-1)  # shape [B,G_h,G_w]
# pred_classes: raw logits of shape [B,G_h,G_w,K]
loss_classes = nn.CrossEntropyLoss()(pred_classes.permute(0,3,1,2), target_class_indices)  # permute to [B,K,G_h,G_w]
```

3) NMS utility (simple):

```python
import torchvision.ops as ops
def nms_boxes(boxes, scores, iou_thresh=0.5):
    # boxes: [N,4] in (xmin,ymin,xmax,ymax), scores: [N]
    keep = ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_thresh)
    return keep.numpy()
```

**Testing and run checklist**

1. Ensure `dataset_dir` variable points to the extracted LabelStudio archive folder.
2. Install dependencies (suggested `requirements.txt` below), then run the notebook cells top-to-bottom.
3. Test `get_iou_bbox` by calling it with two identical boxes — it should return 1.0.
4. Visual check: run the visualization cells for both detection and segmentation outputs (a few examples).
5. Unit checks:
   - After dataset creation, pick a sample and verify that image tensor shape matches `img_height,img_width` and the targets tensor grid corresponds to box positions.
   - For segmentation masks, ensure one-hot channels sum to 1 at each pixel.

**Suggested `requirements.txt`**
```
torch
torchvision
numpy
matplotlib
opencv-python
Pillow
scikit-learn
segmentation-models-pytorch
```

**Next steps (recommended)**
1. Apply the coordinate-scaling fix to `ObjectDetectionDataset` and regenerate `train_dataset` / `val_dataset`.
2. Update `compute_loss` to prepare class indices for `CrossEntropyLoss` or switch to BCE-based loss consistently.
3. Add NMS to object detection evaluation and postprocessing.
4. Run the notebook end-to-end on a machine with GPU (if available). Reduce epochs for quick checks.

If you want, I can apply the coordinate-scaling and loss fixes directly into `HW2.ipynb`, add a `requirements.txt`, and run a quick static check. Which of the next steps should I perform now?

---

**Copy-paste code for notebook cells**

Below are ready-to-paste code snippets for each exercise and for each cell that originally contained "#### YOUR CODE STARTS HERE ####" placeholders. Paste each block into the corresponding cell in `HW2.ipynb` and run.

Exercise 1 — EX1-a (visualize one image and bbox)
#### YOUR CODE STARTS HERE ####
import glob, os, xml.etree.ElementTree as ET

# Update dataset_dir to your extracted LabelStudio folder if needed
dataset_dir = globals().get('dataset_dir', 'dataset')

# Find some image
img_paths = glob.glob(os.path.join(dataset_dir, '**', '*.jpg'), recursive=True) + glob.glob(os.path.join(dataset_dir, '**', '*.png'), recursive=True)
if len(img_paths) == 0:
    raise FileNotFoundError(f"No images found under {dataset_dir}. Update dataset_dir accordingly.")
img_path = img_paths[0]
img = Image.open(img_path).convert('RGB')

# Locate Pascal VOC XML by basename
base = os.path.splitext(os.path.basename(img_path))[0]
xml_candidates = glob.glob(os.path.join(dataset_dir, '**', base + '*.xml'), recursive=True)
boxes = []
if xml_candidates:
    tree = ET.parse(xml_candidates[0])
    root = tree.getroot()
    for obj in root.findall('object'):
        cls = obj.find('name').text
        bnd = obj.find('bndbox')
        xmin = int(bnd.find('xmin').text); ymin = int(bnd.find('ymin').text)
        xmax = int(bnd.find('xmax').text); ymax = int(bnd.find('ymax').text)
        width = xmax - xmin; height = ymax - ymin
        top_left_x, top_left_y = xmin, ymin
        boxes.append((cls, top_left_x, top_left_y, width, height))

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(np.array(img))
if boxes:
    cls, top_left_x, top_left_y, width, height = boxes[0]
    draw_rectangle(ax, top_left_x, top_left_y, width, height)
    ax.set_title(f"{os.path.basename(img_path)} — {cls}")
else:
    ax.set_title(f"{os.path.basename(img_path)} — no XML annotation found")
ax.axis('off')
plt.show()
#### YOUR CODE ENDS HERE ####

Exercise 1 — EX1-b (infer `classes` list)
#### YOUR CODE STARTS HERE ####
import glob, xml.etree.ElementTree as ET, os
dataset_dir = globals().get('dataset_dir', 'dataset')
xml_paths = glob.glob(os.path.join(dataset_dir, '**', '*.xml'), recursive=True)
classes_set = set()
for xp in xml_paths:
    root = ET.parse(xp).getroot()
    for o in root.findall('object'):
        classes_set.add(o.find('name').text)
classes = sorted(list(classes_set)) if classes_set else ['object']
#### YOUR CODE ENDS HERE ####

Exercise 1 — EX1-b (image sizes and grid)
#### YOUR CODE STARTS HERE ####
# Model input size and grid definition
img_width = 416
img_height = 277

grid_height = 4
grid_width = 5
#### YOUR CODE ENDS HERE ####

Exercise 1 — EX1-b (ObjectDetectionDataset implementation)
#### YOUR CODE STARTS HERE ####
import glob, os, xml.etree.ElementTree as ET
from torchvision import transforms as T

class ObjectDetectionDataset(Dataset):
    def __init__(self, dataset_dir, classes, img_width, img_height, grid_height, grid_width):
      super(ObjectDetectionDataset, self).__init__()
      self.dataset_dir = dataset_dir
      self.grid_height = grid_height
      self.grid_width = grid_width
      self.classes = classes
      self.K = len(classes)
      self.img_width = img_width
      self.img_height = img_height
      self.cell_width = np.ceil(self.img_width / self.grid_width)
      self.cell_height = np.ceil(self.img_height / self.grid_height)

      xml_paths = glob.glob(os.path.join(dataset_dir, '**', '*.xml'), recursive=True)
      images = []
      annotations = []
      for xp in xml_paths:
        tree = ET.parse(xp); root = tree.getroot()
        filename_tag = root.find('filename')
        if filename_tag is not None:
          fname = filename_tag.text
          for ext in [fname, fname + '.jpg', fname + '.png']:
            candidate = os.path.join(os.path.dirname(xp), ext)
            if os.path.exists(candidate):
              img_path = candidate
              break
          else:
            img_path = None
        else:
          base = os.path.splitext(os.path.basename(xp))[0]
          candidates = glob.glob(os.path.join(os.path.dirname(xp), base + '.*'))
          img_path = candidates[0] if candidates else None

        if img_path is None:
          continue

        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size

        # Parse bboxes and scale them into model input size
        ann_list = []
        for obj in root.findall('object'):
          cname = obj.find('name').text
          if cname not in self.classes:
            continue
          class_id = self.classes.index(cname)
          bnd = obj.find('bndbox')
          xmin = float(bnd.find('xmin').text); ymin = float(bnd.find('ymin').text)
          xmax = float(bnd.find('xmax').text); ymax = float(bnd.find('ymax').text)
          w = xmax - xmin; h = ymax - ymin
          cx = xmin + w / 2.0; cy = ymin + h / 2.0
          # scale coordinates to model input size
          scale_x = self.img_width / orig_w
          scale_y = self.img_height / orig_h
          cx = cx * scale_x; cy = cy * scale_y
          w = w * scale_x; h = h * scale_y
          ann_list.append([class_id, cx, cy, w, h])

        images.append(pil_img)
        annotations.append(ann_list)

      self.images = images
      self.annotations = annotations
      self.transform = T.Compose([T.Resize((self.img_height, self.img_width)), T.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(image)
        bboxes = self.annotations[idx]
        targets = self.create_targets_tensor(bboxes)

        return image, targets

    def create_targets_tensor(self, annotations):
        targets = torch.zeros((self.grid_height, self.grid_width, self.K + 5))
        for annotation in annotations:
            class_id, x_center, y_center, width, height = annotation
            cell_x_id = int(x_center // self.cell_width)
            cell_y_id = int(y_center // self.cell_height)
            if targets[cell_y_id, cell_x_id,  self.K+4] == 1:
              break
            targets[cell_y_id, cell_x_id, class_id] = 1
            targets[cell_y_id, cell_x_id, self.K:self.K+2] = torch.tensor([
                2*(x_center - (cell_x_id + 1 - 0.5) * self.cell_width) / self.cell_width,
                2*(y_center - (cell_y_id + 1 - 0.5) * self.cell_height) / self.cell_height
            ])
            targets[cell_y_id, cell_x_id,  self.K+2:self.K+4] = torch.tensor([
                width / self.img_width, height / self.img_height
            ])
            targets[cell_y_id, cell_x_id,  self.K+4] = 1
        return targets
#### YOUR CODE ENDS HERE ####

Exercise 1 — EX1-b (create train/val datasets)
#### YOUR CODE STARTS HERE ####
from sklearn.model_selection import train_test_split
dataset_dir = globals().get('dataset_dir', 'dataset')
full_dataset = ObjectDetectionDataset(dataset_dir, classes, img_width, img_height, grid_height, grid_width)
indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

class SubsetDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        img = self.base.images[self.indices[idx]]
        ann = self.base.annotations[self.indices[idx]]
        img = self.base.transform(img)
        targets = self.base.create_targets_tensor(ann)
        return img, targets

train_dataset = SubsetDataset(full_dataset, train_idx)
val_dataset = SubsetDataset(full_dataset, val_idx)
#### YOUR CODE ENDS HERE ####

Exercise 1 — EX1-c (AdvancedObjectDetector)
#### YOUR CODE STARTS HERE ####
import torchvision.models as models

class AdvancedObjectDetector(nn.Module):
  def __init__(self, grid_height=4, grid_width=5, K=2, img_width = 416, img_height = 277):
    super(AdvancedObjectDetector, self).__init__()
    self.grid_height = grid_height
    self.grid_width = grid_width
    self.K = K

    backbone = models.resnet18(pretrained=True)
    feat_dim = backbone.fc.in_features
    self.backbone = nn.Sequential(*list(backbone.children())[:-1])

    out_features = self.grid_height * self.grid_width * (self.K + 5)
    self.head = nn.Sequential(
      nn.Flatten(),
      nn.Linear(feat_dim, out_features)
    )

  def forward(self, x):
    feats = self.backbone(x)
    out = self.head(feats)
    x = out.view(-1, self.grid_height, self.grid_width, self.K + 5)
    x[..., self.K:self.K+2] = torch.tanh(x[..., self.K:self.K+2])
    x[..., self.K+2:self.K+5] = torch.sigmoid(x[..., self.K+2:self.K+5])
    return x
#### YOUR CODE ENDS HERE ####

Exercise 1 — EX1-d (get_iou_bbox)
#### YOUR CODE STARTS HERE ####
def get_iou_bbox(ground_truth_box, predicted_box):
  """
  ground_truth_box: center_x, center_y, width, height
  predicted_box: center_x, center_y, width, height
  """
  gx, gy, gw, gh = ground_truth_box
  px, py, pw, ph = predicted_box

  g_xmin = gx - gw / 2.0; g_xmax = gx + gw / 2.0
  g_ymin = gy - gh / 2.0; g_ymax = gy + gh / 2.0

  p_xmin = px - pw / 2.0; p_xmax = px + pw / 2.0
  p_ymin = py - ph / 2.0; p_ymax = py + ph / 2.0

  inter_xmin = max(g_xmin, p_xmin)
  inter_ymin = max(g_ymin, p_ymin)
  inter_xmax = min(g_xmax, p_xmax)
  inter_ymax = min(g_ymax, p_ymax)

  inter_w = max(0.0, inter_xmax - inter_xmin)
  inter_h = max(0.0, inter_ymax - inter_ymin)
  inter_area = inter_w * inter_h

  g_area = (g_xmax - g_xmin) * (g_ymax - g_ymin)
  p_area = (p_xmax - p_xmin) * (p_ymax - p_ymin)

  union_area = g_area + p_area - inter_area
  iou = inter_area / union_area if union_area > 0 else 0.0
  return iou
#### YOUR CODE ENDS HERE ####

Exercise 2 — EX2-a (read and visualize image + mask)
#### YOUR CODE STARTS HERE ####
import glob, os, json, cv2, numpy as np
dataset_dir = globals().get('dataset_dir', 'dataset')
img_paths = glob.glob(os.path.join(dataset_dir, '**', '*.jpg'), recursive=True) + glob.glob(os.path.join(dataset_dir, '**', '*.png'), recursive=True)
if len(img_paths) == 0:
    raise FileNotFoundError(f"No images found under {dataset_dir}. Update dataset_dir accordingly.")
image_path = img_paths[0]
image = Image.open(image_path).convert('RGB')

coco_json_candidates = glob.glob(os.path.join(dataset_dir, '**', '*.json'), recursive=True)
mask = None
if coco_json_candidates:
    coco = json.load(open(coco_json_candidates[0], 'r'))
    anns_by_fname = {}
    for ann in coco.get('annotations', []):
        img_id = ann['image_id']
        filename = next((im['file_name'] for im in coco.get('images', []) if im['id']==img_id), None)
        if filename:
            anns_by_fname.setdefault(filename, []).append(ann)
    base = os.path.basename(image_path)
    anns = anns_by_fname.get(base, [])
    if anns:
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for ann in anns:
            segs = ann.get('segmentation', [])
            for seg in segs:
                pts = np.array(seg, dtype=np.int32).reshape(-1,2)
                cv2.fillPoly(mask, [pts], color=ann.get('category_id', 1))
if mask is None:
    base_noext = os.path.splitext(os.path.basename(image_path))[0]
    mask_candidates = glob.glob(os.path.join(dataset_dir, '**', base_noext + '*mask*.*'), recursive=True)
    if mask_candidates:
        mask = cv2.imread(mask_candidates[0], cv2.IMREAD_GRAYSCALE)
if mask is None:
    mask = np.zeros((image.height, image.width), dtype=np.uint8)

plt.figure(figsize=(12,8))
plt.subplot(1,2,1); plt.imshow(image); plt.axis('off'); plt.title('Original Image')
plt.subplot(1,2,2); plt.imshow(mask, cmap='viridis'); plt.axis('off'); plt.title('Mask')
plt.show()
#### YOUR CODE ENDS HERE ####

Exercise 2 — EX2-b (SegmentationDataset)
#### YOUR CODE STARTS HERE ####
import glob, os, json, cv2, numpy as np
from torchvision import transforms as T

class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir, classes, img_width, img_height):
      self.dataset_dir = dataset_dir
      self.img_height = img_height
      self.img_width = img_width
      self.classes = classes
      self.K = len(classes)
      self.images = []
      self.masks = []

      coco_json_candidates = glob.glob(os.path.join(dataset_dir, '**', '*.json'), recursive=True)
      if coco_json_candidates:
        coco = json.load(open(coco_json_candidates[0], 'r'))
        images_info = {im['file_name']: im for im in coco.get('images', [])}
        anns_by_fname = {}
        for ann in coco.get('annotations', []):
          img_id = ann['image_id']
          filename = next((im['file_name'] for im in coco.get('images', []) if im['id']==img_id), None)
          if filename:
            anns_by_fname.setdefault(filename, []).append(ann)
        for fname, anns in anns_by_fname.items():
          img_path = os.path.join(dataset_dir, fname)
          if not os.path.exists(img_path):
            base_dir = os.path.dirname(coco_json_candidates[0])
            candidate = os.path.join(base_dir, fname)
            if os.path.exists(candidate):
              img_path = candidate
            else:
              continue
          img = Image.open(img_path).convert('RGB')
          mask = np.zeros((img.height, img.width), dtype=np.uint8)
          # build a mapping from category_id -> contiguous channel index if available
          catid2idx = {c['id']: i+1 for i, c in enumerate(coco.get('categories', []))}
          for ann in anns:
            segs = ann.get('segmentation', [])
            cat = ann.get('category_id', 1)
            ch = catid2idx.get(cat, 1)
            for seg in segs:
              pts = np.array(seg, dtype=np.int32).reshape(-1,2)
              cv2.fillPoly(mask, [pts], color=ch)
          self.images.append(img)
          self.masks.append(mask)
      else:
        img_paths = glob.glob(os.path.join(dataset_dir, '**', '*.jpg'), recursive=True) + glob.glob(os.path.join(dataset_dir, '**', '*.png'), recursive=True)
        for img_path in img_paths:
          base = os.path.splitext(os.path.basename(img_path))[0]
          mask_candidates = glob.glob(os.path.join(dataset_dir, '**', base + '*mask*.*'), recursive=True)
          if not mask_candidates:
            continue
          img = Image.open(img_path).convert('RGB')
          mask = cv2.imread(mask_candidates[0], cv2.IMREAD_GRAYSCALE)
          self.images.append(img)
          self.masks.append(mask)

      self.image_transform = T.Compose([T.Resize((self.img_height, self.img_width)), T.ToTensor()])
      self.mask_transform = lambda m: cv2.resize(m, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        h, w = mask.shape
        one_hot = np.zeros((self.K + 1, h, w), dtype=np.uint8)
        # map cat ids to channels if necessary; by default assume small integer labels
        for i in range(1, self.K + 1):
            one_hot[i] = (mask == i).astype(np.uint8)
        one_hot[0] = (mask == 0).astype(np.uint8)
        one_hot = torch.from_numpy(one_hot.astype(np.float32))
        return image, one_hot

    def parse_annotation(self, annt):
      mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
      return mask
#### YOUR CODE ENDS HERE ####

Exercise 2 — EX2-b (create train/val for segmentation)
#### YOUR CODE STARTS HERE ####
from sklearn.model_selection import train_test_split
seg_dataset = SegmentationDataset(dataset_dir, classes, img_width, img_height)
indices = list(range(len(seg_dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

class SegSubset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

train_dataset = SegSubset(seg_dataset, train_idx)
val_dataset = SegSubset(seg_dataset, val_idx)
#### YOUR CODE ENDS HERE ####

Exercise 2 — EX2-c (instantiate segmentation model)
#### YOUR CODE STARTS HERE ####
import segmentation_models_pytorch as smp
num_classes = len(classes) + 1
segment_model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=num_classes)
segment_model.train()
segment_model = segment_model.to(device)
#### YOUR CODE ENDS HERE ####

Exercise 3 — load test images
#### YOUR CODE STARTS HERE ####
import glob, os
from torchvision import transforms as T
dataset_dir = globals().get('dataset_dir', 'dataset')
image_paths = glob.glob(os.path.join(dataset_dir, 'test', '**', '*.jpg'), recursive=True) + glob.glob(os.path.join(dataset_dir, 'test', '**', '*.png'), recursive=True)
if len(image_paths) == 0:
    image_paths = glob.glob(os.path.join(dataset_dir, '**', '*.jpg'), recursive=True) + glob.glob(os.path.join(dataset_dir, '**', '*.png'), recursive=True)
image_transform = T.Compose([T.Resize((img_height, img_width)), T.ToTensor()])
test_images = [image_transform(Image.open(p).convert('RGB')) for p in image_paths]
#### YOUR CODE ENDS HERE ####

Exercise 3 — postprocess_masks
#### YOUR CODE STARTS HERE ####
def postprocess_masks(bboxes, masks, confidence_threshold=0.35):
  import numpy as np
  boxes = bboxes.copy()
  if isinstance(masks, torch.Tensor):
    masks_np = masks.detach().cpu().numpy()
  else:
    masks_np = masks.copy()
  C, H, W = masks_np.shape
  refined = masks_np.copy()
  grid_h, grid_w = boxes.shape[:2]
  conf_map = boxes[..., K+4]
  ids = np.stack(np.where(conf_map >= confidence_threshold), axis=-1)
  for gy, gx in ids:
    cls_id = int(np.argmax(boxes[gy, gx, :K]))
    rel_bbox = boxes[gy, gx, K:K+4]
    cx, cy, bw, bh = rel_to_abs_coord(rel_bbox, gy, gx, img_width=img_width, img_height=img_height, cell_width=cell_width, cell_height=cell_height)
    xmin = int(max(0, np.floor(cx - bw / 2.0)))
    ymin = int(max(0, np.floor(cy - bh / 2.0)))
    xmax = int(min(W, np.ceil(cx + bw / 2.0)))
    ymax = int(min(H, np.ceil(cy + bh / 2.0)))
    if xmax <= xmin or ymax <= ymin:
      continue
    inside = np.zeros((H, W), dtype=bool)
    inside[ymin:ymax, xmin:xmax] = True
    refined[cls_id, ~inside] = 0
    for c in range(1, C):
      if c != cls_id:
        refined[c, inside] = 0
  foreground_max = np.max(refined[1:], axis=0) if C > 1 else np.zeros((H, W))
  refined[0] = np.clip(1.0 - foreground_max, 0.0, 1.0)
  return refined
#### YOUR CODE ENDS HERE ####

Exercise 3 — inference + visualization loop
#### YOUR CODE STARTS HERE ####
for img in test_images:
  img_batch = img.unsqueeze(0).to(device)
  with torch.inference_mode():
    bboxes = obj_detection_model(img_batch).detach().cpu().numpy()[0]
    pred_masks = torch.nn.functional.softmax(segment_model(img_batch.to(device)), dim=1).detach().cpu()[0]
  refined_masks = postprocess_masks(bboxes, pred_masks, confidence_threshold=0.35)
  refined_masks_t = torch.from_numpy(refined_masks.astype(np.float32))
  visualise_with_masks(img.cpu(), pred_masks, refined_masks_t)
#### YOUR CODE ENDS HERE ####

---

If you want, I can now apply the remaining suggested fixes directly into `HW2.ipynb` (coordinate-scaling fix is already included above), update `compute_loss` to produce class indices, and add an example NMS utility. Tell me which of those you'd like next.
