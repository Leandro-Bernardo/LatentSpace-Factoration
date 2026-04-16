import os
import torch
import numpy as np
import cv2 as cv

from typing import Final, Tuple, Optional
import torchvision.transforms as T
from torch.nn.functional import one_hot


def load_model() -> torch.ScriptModule:
    """Loads a model stored as torch script

    Returns:
        torch.jit: A scripted torch module
    """
    SEGMENTATION_NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "_resources", "TracedSegmentation.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(SEGMENTATION_NETWORK_CHECKPOINT, map_location=device)
    model.eval()
    return model

#  load_model, generate_masks, resize_masks

def predict(model: torch.ScriptModule, image: torch.Tensor) -> torch.Tensor:
    """Uses the informed model to generate the masks from the informed image

    Args:
        model (ScriptModule): a segmentation lightning model
        image (torch.Tensor): a image as tensor of shape [C,H,W]

    Returns:
        torch.Tensor: returns the logit masks as a tensor of shape [N,H,W]
    """
    if len(image.shape) == 3:
        image = image[None, ...]
    masks = model(image)
    masks = masks.detach()
    return masks.squeeze(dim=0)

def prediciton_into_binary_masks(prediction: torch.Tensor) -> torch.Tensor:
    """Transforms the logit masks into binary masks.

    Args:
        prediction (torch.Tensor): logit masks of shape [N,H,W]

    Returns:
        torch.Tensor: returns the masks of shape [N,H,W]
    """
    device = prediction.device
    predicted = prediction.detach().cpu().numpy()
    cl_n, _, _ = predicted.shape
    mult_class = np.stack(list(map(lambda msk: msk, predicted)), axis=0).argmax(axis=0)
    masks = one_hot(torch.from_numpy(mult_class), cl_n).numpy().transpose((2,0,1))
    return torch.tensor(masks, dtype=torch.float32).to(device)

def generate_masks(model: torch.ScriptModule, image: torch.Tensor) -> torch.Tensor:
    """Generates the masks from a segmentation model and a image

    Args:
        model (ScriptModule): a segmentation lightning model
        image (torch.Tensor): a image as tensor of shape [C,H,W]
        resize_shape (Optional[Tuple[int, int]]): a optional tuple of dimentions to resize the masks

    Returns:
        torch.Tensor: returns the binary masks of shape [N,H,W]
    """
    masks = predict(model=model, image=image)
    return prediciton_into_binary_masks(masks)

def resize_masks(masks: torch.Tensor, resize_shape: Tuple[int, int]) -> torch.Tensor:
    """Resizes the masks to the target shape

    Args:
        masks (torch.Tensor): a tensor of shape [N,H,W]
        target_shape (Tuple[int, int]): a tuple of the target shape (H,W)

    Returns:
        torch.Tensor: returns the resized masks of shape [N,H,W]
    """
    return T.Resize(size=resize_shape)(masks)

# Auxiliary functions
def load_image(img_path: str, device: str, resize_shape: Optional[Tuple[int, int]]=None) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Loads a image resizing to the necessary dimentions

    Args:
        img_path (str): the path to the image
        device (str): transports the data to the necessary device
        resize_shape (Optional[Tuple[int, int]]): a optional tuple of dimentions to resize the image

    Returns:
        Tuple[torch.Tensor, Tuple[int, int]]: returns a tuple containing the image as a tensor of shape [C,H,W] and the original shape of the image as a tuple (H,W)
    """
    img = cv.imread(img_path)
    img = np.transpose(img, (2,0,1))
    if resize_shape is not None:
        output_image = T.Resize(size=resize_shape)(torch.as_tensor(img, dtype=torch.float32).to(device))
    else:
        output_image = torch.as_tensor(img, dtype=torch.float32).to(device)
    return output_image, img.shape[1:]

def visualizar_masks_side_by_side(image_bgr: np.ndarray, masks_tensor: torch.Tensor, alpha=0.5, threshold=0.5):
    """
    Visualiza máscaras de segmentação sobrepostas à imagem original e concatena lado a lado.

    Args:
        image_bgr (np.ndarray): Imagem original [H, W, C] (formato OpenCV BGR).
        masks_tensor (torch.Tensor): Máscaras [N, H, W] (pode estar na GPU, float ou bool).
        alpha (float): Transparência da máscara (0.0 a 1.0).
        threshold (float): Limiar para considerar um pixel como máscara (se for probabilidade).

    Returns:
        np.ndarray: Imagem resultante com [Original | Máscara Sobreposta].
    """
    # 1. Copia a imagem para não alterar a original e cria o canvas de sobreposição
    image = image_bgr.copy()
    overlay = image.copy()
    
    # 2. Tratamento do Tensor: Move para CPU, detach e converte para Numpy
    if isinstance(masks_tensor, torch.Tensor):
        masks = masks_tensor.detach().cpu().numpy()
    else:
        masks = masks_tensor # Caso já seja numpy

    # 3. Validação de Shapes
    N, H, W = masks.shape
    img_h, img_w, _ = image.shape
    
    if (H != img_h) or (W != img_w):
        # Opcional: Resize das máscaras se não baterem com a imagem
        print(f"Aviso: Shape da máscara {masks.shape} difere da imagem {(img_h, img_w)}. Redimensionando...")
        masks = np.array([cv.resize(m, (img_w, img_h)) for m in masks])

    # 4. Aplicação das Máscaras
    # Gera N cores aleatórias para as N máscaras
    np.random.seed(42) # Seed para cores consistentes
    colors = np.random.randint(0, 255, size=(N, 3), dtype=np.uint8)

    for i in range(N):
        mask = masks[i]
        
        # Converte para binário baseado no threshold
        binary_mask = mask > threshold
        
        if not np.any(binary_mask):
            continue
            
        color = colors[i].tolist() # BGR
        
        # Pinta a região da máscara na imagem 'overlay'
        # Usamos indexação booleana para pintar apenas onde a máscara existe
        overlay[binary_mask] = color
        
        # Opcional: Desenhar contorno para melhor definição
        contours, _ = cv.findContours(binary_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(overlay, contours, -1, color, 2)

    # 5. Aplica o efeito de transparência (Alpha Blending)
    # cv.addWeighted faz: src1 * alpha + src2 * beta + gamma
    masked_image = cv.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # 6. Concatena lado a lado (Horizontal Stack)
    final_result = np.hstack((image, masked_image))
    
    return final_result

if __name__ == "__main__":
    IMG = "Y://repo//Materiais Dispersos Totais - Testes de Injecao//Solidos//A25//0a3305a2-1fb8-41c7-8962-1d0760445eee.jpg"
    
    model = load_model()
    img, ori_shape = load_image(img_path=IMG, device=next(model.parameters()).device, resize_shape=(1024, 512))
    
    masks = generate_masks(model=model, image=img)
    
    ori_sized = T.Resize(size=ori_shape)(img)
    result_mask = visualizar_masks_side_by_side(ori_sized.detach().cpu().numpy().transpose((1,2,0)).astype(np.uint8), masks)
    
    cv.imwrite("resultado_final.jpg", result_mask)
    pass