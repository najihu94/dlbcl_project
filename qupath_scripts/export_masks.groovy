import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject
import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

// Get the main QuPath data structures
def imageData = getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()
// Request all objects from the hierarchy & filter only the annotations
def annotations = hierarchy.getAnnotationObjects()
// Define downsample value for export resolution & output directory, creating directory if necessary
def downsample = 1.0
def pathOutput = buildFilePath(QPEx.PROJECT_BASE_DIR, 'masks', String.format('%s',server.getMetadata().getName()))
mkdirs(pathOutput)
// Define image export type; valid values are JPG, PNG or null (if no image region should be exported with the mask)
// Note: masks will always be exported as PNG
def imageExportType = 'PNG'

// Export each annotation
annotations.each {
    saveImageAndMask(pathOutput, server, it, downsample, imageExportType)
}

def path = 'C:/Users/Hussi/Desktop/Promotion/Annotationen/Qupath/HE/combined masks/'
annotations.each {
    saveImageAndMask2(path, server, it, downsample, imageExportType)
}

/**
 * Save extracted image region & mask corresponding to an object ROI.
 *
 * @param pathOutput Directory in which to store the output
 * @param server ImageServer for the relevant image
 * @param pathObject The object to export
 * @param downsample Downsample value for the export of both image region & mask
 * @param imageExportType Type of image (original pixels, not mask!) to export ('JPG', 'PNG' or null)
 * @return
 */
def saveImageAndMask(String pathOutput, ImageServer server, PathObject pathObject, double downsample, String imageExportType) {
    // Extract ROI & classification name
    def roi = pathObject.getROI()
    def pathClass = pathObject.getPathClass()
    def classificationName = pathClass == null ? 'None' : pathClass.toString()
    if (roi == null) {
        print 'Warning! No ROI for object ' + pathObject + ' - cannot export corresponding region & mask'
        return
    }
    // Create a region from the ROI
    def region = RegionRequest.createInstance(server.getPath(), downsample, roi)
    // Create a name
    String name = String.format('%s_(%d,%d,%d,%d)',
   //         server.getMetadata().getName(),
            classificationName,
            // region.getDownsample(),
            region.getX(),
            region.getY(),
            region.getWidth(),
            region.getHeight()
    )

    // Request the BufferedImage
    def img = server.readBufferedImage(region)

    // Create a mask using Java2D functionality
    // (This involves applying a transform to a graphics object, so that none needs to be applied to the ROI coordinates)
    def shape = RoiTools.getShape(roi)
    
    //ersetzen, wenn nur Zellregion gewünscht ist!
    // def imgMask = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_BYTE_GRAY)  
    def imgMask = new BufferedImage(512, 512, BufferedImage.TYPE_BYTE_GRAY)
    
    def g2d = imgMask.createGraphics()
    g2d.setColor(Color.WHITE)
    g2d.scale(1.0/downsample, 1.0/downsample)
    
    //einkommentieren, wenn nur Zellregion gewünscht ist!
    // g2d.translate(-region.getX(), -region.getY()) 
  
    g2d.fill(shape)
    g2d.dispose()
    
    // Export the mask
    def fileMask = new File(pathOutput, name + '_mask.png')
    ImageIO.write(imgMask, 'PNG', fileMask)

}
print 'Done!'

