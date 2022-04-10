<h1>Training Data</h1>

<h3>Audio Source</h3>
<p><a href="https://drive.google.com/drive/folders/1jtSzfG69MFptyQGHKbSkFLnSg_il9D50">Link to Audiofiles</a></p>

<h3>DataSets</h3>
<p><a href="https://drive.google.com/drive/folders/1To_0V9DeWv4i3t7ReOIT862S_xebgu0y">Link to DataSets</a></p>

<h3>Plots</h3>
<p><a href="https://drive.google.com/drive/folders/164V23e5c0ud7w_xzMHN-FT-VKh17vg8a">Link to Plots</a></p>

<h2>Aufbau der Trainingsdaten</h2>

Da als erste Challenge die erstellung von geeigneten Trainingsdaten ist, müssen im ersten Schritt Audio-Samples erstellt werden. Somit können keine im Internet zur Verfügung stehenden Datansätze genutzt werden.
Aus diesem Grund ist besteht die Herausforderung darin, die optimale Länger von Samples herauszufinden, sowie das Klassifizieren der Samples.
Zur Klassifizierung der Daten haben wir uns für Unsupervised Learning durch das Kmeans-Clustering entschiedne. Durch dieses Verfahren werden Cluster gebildet, welche aus Audio-Samples, die sich ähnlich sind, gebildet. Alle Audio-Samples die zum selben Cluster gehören, erhalten auch die selbe Kennzeichnung (Label).
  
Damit wir unsere Samples klassifizieren konnten, mussten wir verschieden Audio-Features ermitteln. Der umgesetzte Prozess zur Erstellung unserer Trainingsdaten ist in drei Schritte aufgeteil:
  
  1. Laden des von uns ausgewählten Tracks (ca. 1 Stunde lang, ca. 700mb groß) und zerschneiden in Samples.
  2. Jedes neu erstellte Sample laden und für dieses die Audio-Features ermitteln.
  3. Anhand der Features Cluster bilden und die Samples somit Klassifizieren.
  
Sämtlich Daten die innerhalb dieser drei Prozessschritte entstehen, werden in ein Dataframe eingetragen und als CSV-Datei exportiert. 
Durch diesen Prozess haben wir mehrere Datasets von Audio-Samples in unterschiedlichen Längen und Klassifizierungen gebildet, die wir in Challenge zwei verwenden werden. Sämtliche von uns erstellte Datasets finden sich in unserem Google-Drive und können über die oben stehenden Links aufgerufen werden.
  
Die aus den Samples ausgelesenen Features sind:
  
![grafik](https://user-images.githubusercontent.com/99210485/162636440-50d69b29-3d42-499b-a354-f0a732c516b5.png)

Der Aufbau unserer Datasets sieht wie folgt aus:

![grafik](https://user-images.githubusercontent.com/99210485/162636754-f01fefbe-1680-4a5b-9e00-5efc8d99137b.png)


Falls wir im laufe der Challenge 2&3 festellen, dass wir unsere Daten anpassen müssen, können wir das innerhalb unseres Programmcodes durch das verändern von zwei Parametern schnell erledigen.
  
  
