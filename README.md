# cropper
## Структура файлов
Исходный датасет должен быть в одной дирректории, но может быть в разных подпапках:
<pre>
Path_to_dataset/  
   --annotation/  
     --img1_annotation
     --img2_annotation
     ...
   --image/
     --img1
     --img2
     ...
</pre>
Изображения в дирректории ищутся рекурсивно, что означает, что кроме изображений и аннотаций к ним в дирректории  
ничего не должно находиться с искомым расширением, в том числе и в подкаталогах.

## Устарело:
Исходный датасет должен быть в одной дирректории:  
<pre>
Path_to_dataset/  
   --img1.extantion  
   --img1_annotation.extantion  
   --img2.extantion  
   --img2_annotation.extantion  
   ...  
 *extantion - расширение файла (png, jpg и т.д.) </pre>
