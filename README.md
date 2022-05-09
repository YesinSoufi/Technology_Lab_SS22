# <h1>Technology_Lab_SS22</h1>

<p>Elena Müller 42616</p>
<p>Osman Kaplan 42581 </p>
<p>Niklas Öxle 42614 </p>
<p>Sascha Lehmann 42599 </p>
<p>Yesin Soufi 42612 </p>
<p>Jakob Schaal 42613 </p>

Aufgabenverteilung: [Kanban-Board zum Projekt](https://github.com/YesinSoufi/Technology_Lab_SS22/projects/2)

<h2>Projektbeschreibung</h2>

Im Rahmen des Projekts Technology Lab wird ein System entwickelt, das mithilfe eines neuronalen Netzwerkes anhand unbekannter Samples einen Musikstream generiert . Das Projektvorgehen ist in drei Schritte gegliedert:

1. Maschinelles Lernen von Sequenzen
    * Herstellung von Trainingsdaten durch Zerlegung von Musikstücken in Samples
    * Schaffen einer geeigneten Infrastruktur
2. Rekonstruktion des ursprünglichen Tracks bzw. Bilden von Sequenzen adaptierbarer Musik auf Basis bekannter Samples
    * Zusammensetzen (Resampling) der zuvor zerstückelten Musik
3. Rekombination von Musikstreams bzw. Lernen von Sequenzen von adaptierbarer Musik auf Basis zuvor unbekannter Samples
    * Resampling von Samples, die dem Neuronalen Netz zuvor nicht zugeführt  wurden
Im Fokus des Projekts steht neben Entwicklung eines lauffähigen Prototyps v.a. der wissenschaftliche Erkenntnisgewinn.

<h3>Forschungsfragen:</h3> 
Ziel des Projekts ist die Beantwortung der folglich definierten Forschungsfragen. 

   * Lässt sich ein „hörbarer“ (d.h. kaum von einem manuell gemixten Set unterscheidbarer) Stream durch das Zerschneiden und anschließende Resampling von Tracks auf Basis des Maschinellen Lernens erzeugen? 
   * Eignen sich gleichlange Samples oder stochastisch gewählte Samples besser als Datengrundlage? Wie lange ist eine optimale Sampledauer?
   * Wie ähnlich müssen sich die Ursprungstracks bzgl. Stimmung, Genre etc. sein? 
   * Müssen Metadaten wie Lautstärke, Tempo, Spektrum, etc. mit einbezogen werden?

<h4>Systemanforderungen:</h4> 
Zusätzlich zu den vorgegebenen Projektschritten (Challenges) und den sich daraus ableitenden Systemanforderungen wurden Personas und User Stories erarbeitet. 

  * [Personas](https://github.com/YesinSoufi/Technology_Lab_SS22/blob/main/Personas.md)
  * [User Stories](https://github.com/YesinSoufi/Technology_Lab_SS22/blob/main/UserStories.md)

 

<h4>Systementwurf:</h4> 

  * Scenarios and Glossar
 * [ComponentDiagram](https://github.com/YesinSoufi/Technology_Lab_SS22/blob/main/ComponentDiagram.md)
  * [Machine learning Konzept](https://github.com/YesinSoufi/Technology_Lab_SS22/blob/main/assets/MachineLearningKonzept.pdf)
  * [Training data](https://github.com/YesinSoufi/Technology_Lab_SS22/blob/main/TrainingData.md)

<h5>Aktuell benutzte Libraries</h5> 

 * pandas
 * random
 * createSamples
 * feature_extraction
 * cluster_dataset
 * numpy
 * scipy
 * matplotlib.pyplot
 * sklearn
 * librosa
 * IPython
 * glob
 * pydub
 * pathlib


<h3>Wissenschaftliche Basis</h3>

* musicinformationretrieval.com
* https://www.statworx.com/content-hub/blog/einfuehrung-tensorflow/
* http://www.deeplearningbook.org
* https://openai.com/blog/jukebox/
* https://magenta.tensorflow.org/
* https://github.com/librosa/librosa

Regression vs. Klassifizierung 
* Regression: Bei der Regression soll eine kontinuierliche Zielvariable, also unbekannte Werte, vorhergesagt werden (2)(3). 
* Klassifizierung: 

Feed-forward vs. rekursive Netze
* Feed-forward Netze
    * Cnn 
	Musikstücke & Label nach genre
	Ungehobelte samples aus den Genres
	Neuronales netz sollen ungelabelten klassifiziert werden
* Rekursive Netze
    * Lstm

Supervised vs. unsupervised Training
* Supervised:
    * der ideale Ausgabewert ist spezifiziert 
* Unsupervised:
    * Keine idealen Ausgabewerte vorgegeben
    * Das neuronale Netz lernt es die Eingabedaten in eine Reihe von Gruppen einzuordnen, die durch die Ausgabe Neuronen definiert sind (5)


1) Buxmann, P. und Schmitt, H. (2019), Künstliche Intelligenz – Mit Algorithmen zum wirtschaftlichen Erfolg, Darmstadt 2019
2) Marsland, S. (2015), Machine Learning – An Algorithmic Perspective, 2. Auflage, Boca Raton, 2015
3) Murphy, K. P. (2012), Machine Learning – A Probabilistic Perspective, Cambridge 2012
4) https://www.tensorflow.org/tutorials/structured_data/time_series
5) https://p300.zlibcdn.com/dtoken/a42592737c52e134a610d6a57cf4e039/Artificial%20Intelligence%20for%20Humans%2C%20Volume%203%20Deep%20Learning%20and%20Neural%20Networks%20%28Jeff%20Heaton%29%20%28z-lib.org%29.pdf


<h3>Challenge 1</h3> 
**[CreateSamples](https://github.com/YesinSoufi/Technology_Lab_SS22/blob/main/assets/Music_Resampler.pdf)




<h3>Challenge 2</h3> 
**[Prototyp](https://github.com/YesinSoufi/Technology_Lab_SS22/blob/main/assets/Music_Resampler.pdf)

<h3>Verfolgte Ansätze</h3> 
* Zeitreihen-Vorhersage (Regression)
Im ersten Ansatz wurde (auf Empfehlung) eine Zeitreihen-Vorhersage mittels Regression verfolgt. Bei der Regression soll eine kontinuierliche Zielvariable, also unbekannte Werte, vorhergesagt werden (2)(3). Zur Vorhersage wird der Einfuss einzelner Variablen bzw. Features auf die Ausgangsvariable untersucht. Folglich wurden den Audiodaten zunächst Features extrahiert. Zu diesen Features zählen u.a. Root Mean Square Error, Chroma, Spectral Centroid, Zero Crossing Rate und Mel Frequency Cepstral Coefficients. Das Vorgehen lehnt sich an einen Ansatz zur Vorhersage von Wetterdaten an (4). Analog zu den Wetterdaten können die wellenförmigen Audiodaten zunächst aufbereitet und sowohl für Single-step als auch für Multi-step Modelle genutzt werden.  
Der Ansatz wurde verworfen, da aus Challenge 2 die Anforderung hervorgeht, dass bestehende Samples wiederverwendet werden sollen. Der Ansatz eignet sich demnach nicht zur Lösung, da dieser neue Samples kreieren würde. 
