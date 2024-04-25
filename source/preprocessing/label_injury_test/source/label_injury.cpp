#include <iostream>
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkBinaryFillholeImageFilter.h"

using namespace std;
using namespace itk;

int main(int argc, char* argv[])
{
 using pixelType = float;
 
using imageType = Image<pixelType, 3>;

// Intentiate reader and writer types for the input and output images
using readerType = ImageFileReader<imageType>;
using writerType = ImageFileWriter<imageType>;

// Intentiate iterator type
using iteratorType = ImageRegionIterator<imageType>;

// Read the input image
readerType::Pointer reader = readerType::New();
reader->SetFileName(argv[1]);
reader->Update();

// Allocate the output image
imageType::Pointer outputImg = imageType::New();
outputImg->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
outputImg->SetSpacing(reader->GetOutput()->GetSpacing());
outputImg->SetOrigin(reader->GetOutput()->GetOrigin());
outputImg->SetDirection(reader->GetOutput()->GetDirection());
outputImg->Allocate();

// Allocate the brain mask image
imageType::Pointer maskImg = imageType::New();
maskImg->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
maskImg->SetSpacing(reader->GetOutput()->GetSpacing());
maskImg->SetOrigin(reader->GetOutput()->GetOrigin());
maskImg->SetDirection(reader->GetOutput()->GetDirection());
maskImg->Allocate();

// Assign an iterator to the input images and output images
iteratorType iterInputImg(reader->GetOutput(), reader->GetOutput()->GetLargestPossibleRegion());
iteratorType iterOutputImg(outputImg, outputImg->GetLargestPossibleRegion());
iteratorType iterMaskImg(maskImg, maskImg->GetLargestPossibleRegion());

// Modify the label of the delineations
for(iterInputImg.GoToBegin(), iterMaskImg.GoToBegin(); !iterInputImg.IsAtEnd(), !iterMaskImg.IsAtEnd(); ++iterInputImg, ++iterMaskImg)
{
		if(iterInputImg.Get() != 0 )
			iterMaskImg.Set(1);
}

//Fill the hole inside the brain delineation
using binaryFillholeFilterType = BinaryFillholeImageFilter<imageType>;
binaryFillholeFilterType::Pointer binaryFillhole = binaryFillholeFilterType::New();

binaryFillhole->SetInput(maskImg);
binaryFillhole->SetForegroundValue(1);
binaryFillhole->FullyConnectedOn();
binaryFillhole->Update();
imageType::Pointer fillMaskImg = binaryFillhole->GetOutput();
iteratorType iterFillMaskImg(fillMaskImg, fillMaskImg->GetLargestPossibleRegion());


// Label the injury
for(iterInputImg.GoToBegin(), iterOutputImg.GoToBegin(), iterFillMaskImg.GoToBegin(); !iterInputImg.IsAtEnd(), !iterOutputImg.IsAtEnd(), !iterFillMaskImg.IsAtEnd(); ++iterInputImg, ++iterOutputImg, ++iterFillMaskImg)
{
		if(iterInputImg.Get() == 0 && iterFillMaskImg.Get() == 1 )
			iterOutputImg.Set(7);
		else
			iterOutputImg.Set(iterInputImg.Get());
			
}


// Write on the hard disk the output image
writerType::Pointer writer = writerType::New();
writer->SetFileName(argv[2]);
writer->SetInput(outputImg);
writer->Update();

return 1;
}