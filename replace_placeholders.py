import os
import shutil

placeholder_size = 2469
supp_img = r'C:\Users\aashi\.gemini\antigravity-ide\brain\73ef348e-b146-4169-9a29-8fa5f0e86f5d\generic_supplement_1787138989185.png'
disease_img = r'C:\Users\aashi\.gemini\antigravity-ide\brain\73ef348e-b146-4169-9a29-8fa5f0e86f5d\generic_disease_1787139480980.png'

def replace_placeholders(folder, replacement_img):
    count = 0
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.getsize(path) == placeholder_size:
            shutil.copy(replacement_img, path)
            count += 1
    print(f"Replaced {count} images in {folder}")

replace_placeholders('static/supplement_images', supp_img)
replace_placeholders('static/disease_images', disease_img)
