Set annotationMeasurements = []
getAnnotationObjects().each{
it.getMeasurementList().getMeasurementNames().each{
annotationMeasurements << it}}
annotationMeasurements.each{ removeMeasurements(QuPath.lib.objects.PathCellObject, it);}

boolean prettyPrint = false 
// false results in smaller file sizes and thus faster loading times, at the cost of nice formating
def gson = GsonTools.getInstance(prettyPrint)
def annotations = getAnnotationObjects()
def name = getProjectEntry().getImageName() + '.json'
def path = buildFilePath(PROJECT_BASE_DIR, 'annotation results')

mkdirs(path)
path = buildFilePath(path, name)
saveAnnotationMeasurements(path)

File file = new File(path)
 file.withWriter('UTF-8') {
     gson.toJson(annotations,it)
 }
 
 print annotations