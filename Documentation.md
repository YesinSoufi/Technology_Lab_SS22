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

<h2>Forschungsfragen</h2> 
Ziel des Projekts ist die Beantwortung der folglich definierten Forschungsfragen. 

   * Lässt sich ein „hörbarer“ (d.h. kaum von einem manuell gemixten Set unterscheidbarer) Stream durch das Zerschneiden und anschließende Resampling von Tracks auf Basis des Maschinellen Lernens erzeugen? 
   * Eignen sich gleichlange Samples oder stochastisch gewählte Samples besser als Datengrundlage? Wie lange ist eine optimale Sampledauer?
   * Wie ähnlich müssen sich die Ursprungstracks bzgl. Stimmung, Genre etc. sein? 
   * Müssen Metadaten wie Lautstärke, Tempo, Spektrum, etc. mit einbezogen werden?

<h2>Wissenschaftliche Grundlagen</h2> 
<h3>Künstliche Neuronale Netzwerke</h3> 

Ein Neuronales Netz besteht aus einer Vielzahl miteinander verbundener künstlicher Neuronen, welche üblicherweise in Schichten (Layers) angeordnet werden. Dabei bezeichnet man die Schicht, welche die Eingabedaten erhält, als Input Layer und die Schicht, die die Ausgabe erzeugt als Output Layer.. Zwischen Input und Output Layer können sich beliebig viele weitere Schichten – sogenannte Hidden Layers – befinden. Jedes Neuron erzeugt nun eine individuelle Ausgabe, die abhängig von der jeweiligen Eingabe und der internen Aktivierungsfunktion des Neurons ist. Dabei werden alle Ausgaben einer Schicht als Eingaben für die jeweils nächste Schicht weitergeleitet, wobei jede Verbindung die Informationen unterschiedlich stark weiterleiten kann. Die Stärke jeder einzelnen Verbindung wird beim Erzeugen und Trainieren des Neuronalen Netzes durch einen speziellen Lernalgorithmus (Back-Propagation) festgelegt. (Müller, 2021)
Künstliche Neuronale Netze besitzen mehrere Perzeptronen/ Neuronen auf jeder Schicht. Sie werden auch Feed-Forward Neural Network bezeichnet, da Eingaben nur in Vorwärtsrichtung verarbeitet werden.

<h3>CNN</h3> 

Das Convolutional Neural Network ist eine spezielle Form eines neuronalen feedforward Netzes. Es bezieht seinen Namen von dem mathematischen Prinzip der Faltung (eng. Convolution). (Frochte, 2020)
Das CNN ist besonders für die Erkennung von Bild- und Audiodaten geeignet. Die durch die Faltung entstehende Reduzierung der Daten erhöht die Geschwindigkeit der Berechnung und bewirkt somit eine Verkürzung des Lernprozesses.
Mehrschichtig vorwärts gekoppelte Netze?
CNNs sind folgendermaßen aufgebaut:
1. Convolutional Layer
2. Pooling Layer
3. Flatten Layer
4. Drop Out Layer
5. Dense Layer

Folgend sind einige bekannte und moderne CNN Architekturen aufgelistet:
* LeNet-5
* AlexNet CNN
* VGG-16
* ResNet

<h3>LSTM</h3> 

Beim LSTM handelt es sich um eine Sonderform des RNN. Long Short-Term Memory Netzwerke (LSTM) sind RNN, die durch ihre Architektur das Verschwinden des Gradienten verhindern und in der Lage sind Langzeitabhängigkeiten zu erfassen. LSTM verwenden Zustandszellen: Diese können mit ihrer wesentlich komplexeren Struktur den Informationsfluss regulieren. Die Zustandszellen von LSTM gewährleisten das Bestehen von Informationen über mehrere Zeitschritte hinweg.
Der Zustand einer Zelle wird über zwei Gates beeinflusst: dem Forget Gate und dem Input Gate. Das Forget Gate ermöglicht es einerseits nicht mehr benötigte Informationen zu löschen und andererseits wichtige Informationen unverändert bestehen zu lassen. Das Input Gate ermöglicht es neue Informationen einzuspeichern. Ein drittes Gate ist das Output Gate, welches die Aktivierung der LSTM Zelle unter Berücksichtigung des Zellzustands ermittelt.

<h3>Dense Layer</h3> 

Beim Dense Layer handelt es sich um die am häufigsten verwendete Schicht in künstlichen neuronalen Netzen. Der Dense Layer wird auch als Fully Connected Layer bezeichnet. Es handelt sich hier um eine Standardschicht, bei der alle Neuronen mit sämtlichen Inputs und Outputs verbunden sind. In dem letzten Dense Layer findet die endgültige Klassifizierung statt.Dense Layer sind eng mit der vorangehenden Schicht verbunden. Neuronen des Dense Layers sind mit jedem Neuron der vorangehenden Schicht verbunden. Hier findet Matrix-Vektor Multiplikation statt. Die Dense Layer übernimmt die Aufgabe der Klassifizierung in einem CNN. 

<h3>Convolutional Layer</h3> 
Convolutional Layer:  Dieser Layer implementiert verschiedene und integriert diese im neuronalen Netz. Dabei wird eine Faltungsmatrix (Kernel) über die Pixelwerte gelegt. Die Gewichte der Kernel sind jeweils unterschiedlich dimensioniert. Durch die Verrechnung mit den Eingabewerten können unterschiedliche Merkmale (Kanten und Features) extrahiert werden.
<h3>Pooling Layer</h3> 
Dieser Layer dient dazu, die Daten besser verallgemeinern zu können. Durch das Pooling Layer (Max Pooling) werden die stärksten Merkmale weitergeleitet. Von den Dimensionen der Eingabedaten, bei Bildern beispielsweise die Anzahl der Pixel, ist abhängig, wie viele Pooling Layer angewandt werden können. Letztendlich ist der Zweck dieser Layer v.a. die Datenreduktion (Jeong, 2021). 
<h3>Flatten Layer</h3> 
Beim Flattening werden die Daten in ein 1-dimensionales Array konvertiert, um sie in die nächste Schicht einzugeben. Die Ausgabe der Faltungsschichten wird geglättet, um einen einzigen langen Merkmalsvektor zu erstellen. (Jeong, 2021)
<h3>Dropout Layer</h3> 
Dropout ist eine Technik, um das Overfitting eines neuronalen Netzes zu reduzieren (Srivastava, 2014). Es kann vorkommen, dass sich neuronale Netzwerke zu stark auf einen oder mehrere Eingabeparameter beschränken. Aus diesem Grund kann Drop Out verwendet werden, sodass bestimmte Verbindungen der Eingangsdaten nicht mehr weitergegeben werden. So wird erreicht, dass sich das Netz nicht zu stark auf einen bestimmten Wert verlässt und unabhängig von einer bestimmten Verbindung einen geeigneten Zusammenhang findet.
<h3>Encoder/Decoder</h3> 
Die Encoder-Decoder-Architektur kann Eingaben und Ausgaben verarbeiten, die beide Sequenzen variabler Länge sind, und eignet sich daher für Sequenztransduktionsprobleme wie die maschinelle Übersetzung. Der Kodierer nimmt eine Sequenz variabler Länge als Eingabe und wandelt sie in einen Zustand mit einer festen Form um. Der Decoder bildet den kodierten Zustand einer festen Form auf eine Sequenz variabler Länge ab.
<h3>Aktivierungsfunktion</h3> 

Durch die Aktivierungsfunktion erhält das neuronale Netz eine nichtlineare Eigenschaft. Dies wird benötigt, um einem neuronalen Netz die Fähigkeit zur Modellierung komplexerer Daten zu bieten. In neuronalen Netzen werden drei verschiedene Funktionen genutzt. Die Wahl der Funktion hängt vom entsprechenden Anwendungsfall ab. Die drei gängigsten Aktivierungsfunktionen werden nachfolgenden näher beschrieben. (Oppermann, 2022)

Relu: Die ReLu-Funktion liefert als Rückgabe den Wert 0 oder 1. Problem negative Werte zählen als 0. ReLU  hat gezeigt, dass die Konvergenz des Gradientenabstiegs in Richtung des globalen Minimums der Verlustfunktion im Vergleich zu anderen Aktivierungsfunktionen beschleunigt. Dies ist auf seine lineare, nicht sättigende Eigenschaft zurückzuführen. Ein Vorteil der RelU-Aktivierungsfunktion ist. dass sie deutlich weniger rechenintensiv als die Sigmoid/tanH-Aktivierungsfunktion ist. Ein Nachteil der RelU-Aktivierungsfunktion hingegen kann sein, dass Werte unter Null als Null gewertet werden. (Oppermann, 2022)



Sigmoid: Rückgabewert im Interval von 0 bis 1. Im Gegensatz zur Relu-Aktivierungsfunktion sind hier auch Werte wie z.B. 0,234 eine gültige Eingabe. (Oppermann, 2022)

tanH: Die TanH-Aktivierungsfunktion liefert als Rückgabewert eine reelle Zahl zwischen [-1,1]. In der Praxis wird die TanH-Aktivierungsfunktion in der Regel der Sigmoid Funktion vorgezogen. Grund hierfür ist, dass die TanH-Funktion im im Gegensatz zur Sigmoid-Funktion Null-zentriert ist. (Oppermann, 2022)


<h2>Challenge 2</h2> 
<h3>Ansatz Dense Network</h3> 
<h4>Vorgehensweise</h4> 
Für Challenge 2 wurde ein Klassifikations-Ansatz gewählt. Hierzu wurde ein Dense Network genutzt. Im ersten Schritt wurden Label Parameter gesetzt. Für die Klassifikation wurden 2 csv-Dateien verwendet (train_file und val_file). Im Trainingsdatensatz wurden die Samples mit einer nach der Reihenfolge gesetzten ID des Songs gelabelt. Anschließend wurde ein neuer Song genommen, der nicht im Trainingsdatensatz enthalten war und  in 8-Sekunden Stücke geschnitten, mit dem Ziel für jedes Sample das zugehörige Label zu prognostizieren. Im Letzten Schritt wurden die Samples nach den prognostizierten Samples sortiert und exportiert.
<h4>Fazit Challenge 2</h4> 
Mit Hilfe  des Dense Networks wurde ein Ergebnis aus Samples generiert. Der generierte Song war hörbar. Jedoch ist dies aufgrund des fehlenden Erinnerungsvermögens aus Zufallsergebnis zu verzeichnen, Die Länge von 8 Sekunden der einzelnen Samples und die daraus resultierende niedrige Anzahl an Cuts begünstigt zudem ein gutes Ergebnis. Als Key Learning aus Challenge 2 wurde deshalb die Notwendigkeit eines Neuronales Netzes, welches Erinnerungsvermögen besitzt, identifiziert. Der Prozess muss durch Machine Learning funktionieren und nicht durch eigens entwickelte Algorithmen und Methodiken. Außerdem wurde uns bewusst, dass ein tieferes Verständnis für neuronale Netze unumgänglich ist und in Vorbereitung auf Challenge 3 aufgebaut werden muss. 

<h2>Challenge 3</h2> 
<h3>Ansatz 1 Encoder/Decoder</h3> 
Die Grundidee dieses Ansatzes besteht darin ein Neuronales Netz unsupervised mit Samples zu trainieren, sodass es sich dabei aussagekräftige Features aneignet. Via Autoencoder wird unsupervised Learning durchgeführt, um Features zu extrahieren. Dabei dient der Decoder nur der Evaluation, um die Qualität unserer extrahierten Features zu prüfen. Danach wird die Similarity zu den vorherigen Samples bestimmt. Die Idee dabei ist, dass die letzten X (z.B. 3) Samples betrachtet werden, um den bisherigen Songverlauf miteinzubeziehen.
<h4>Fazit</h4> 
Es lässt sich zusammenfassen, dass es sich bei dieser Lösung um einen sehr “händischen” Ansatz handelt. Vieles erfolgt per eigens programmierten Algorithmen, wobei sich die Frage stellt was die Maschine macht. Letztendlich müsste nach der Feature-Extraktion ein Wechsel zum Supervised Ansatz erfolgen, da eine händische Labelung nötig gewesen wäre.

<h3>Ansatz 2 -CRNN</h3> 
Die Grundidee des zweiten Ansatzes besteht darin gelabelte Daten zur Klassifizierung geeigneter Übergänge zu nutzen (Supervised Learning). Im Rahmen einer Sequence Classification werden Zeitreihen zur Klassifikation zwischen 0 und 1 betrachtet. 0 bedeutet schlechter Übergang, bei 1 stellt die Maschine einen guten Übergang fest. Um ein Erinnerungsvermögen zu nutzen, werden vorangegangene Daten miteinbezogen. 
<h4>Architektur</h4> 

Es wird ein Erinnerungsvermögen benötigt, da die Datenpunkte eine Beziehung zu den vorherigen Datenpunkten haben. Zur Umsetzung dieses Ansatzes wurde eine Kombination aus CNN und LSTM gewählt. Ein reines RNN wäre leistungstechnisch nicht umsetzbar. CNNs haben hervorragende Auswirkungen auf Daten, die nicht neu angeordnet werden können oder bei denen Elemente verloren gehen, und Recurrent Neural Networks (RNNs) können die Daten so anordnen, dass sie der menschlichen semantischen Beschreibung nahekommen. (Cheng, 2021) Daher kombinieren wir die beiden und nutzen die Vorteile beider, um das Modell zu erstellen. 

Zur Umsetzung dieses Ansatzes werden vier verschiedene Layer-Arten verwendet:

* Convolutional Layer
    * Verwendung von einer Convolutional Layer(1D)
    * Verwendung der Aktivierungsfunktion “ReLu”
    * Datenreduktion 

* Pooling Layer
    * MaxPooling Operation
    * Reduziert Trainingsdauer und Rechenkosten

* Dropout Layer
    * Vermeidung von Overfitting
    * Reduzierung der Neuronen um 25%

* LSTM
    * Berechnet Feature-Vektor aus CNN Output
    * Betrachtung von Zeitreihen

* Dense Layer
    * Klassifiziert LSTM-Output
    * Output von 0 bis 1


<h4>Trainingsdaten</h4> 

Zum Erzeugen von Trainingsdaten wurden Songs in Samples zerschnitten, wobei jedes Sample eine Länge von 3 Sekunden hat. Zwei aufeinander folgende Samples werden als passend gelabeled. Außerdem wurden extra Samples mit Extremfällen integriert.
Im Unterschied zu Challenge 2 soll nun verhindert werden, dass die Input-Songs wieder zusammengeführt werden. 

<h4>Training & Evaluation</h4> 

Für ein gutes Ergebnis ist die Anzahl der Epochen von zentraler Bedeutung, zu wenig Epochen verhindern einen guten Lerneffekt. Bei zu vielen Epochen verliert das Modell die Generalisierung-
kapazität durch Überanpassung. Zum Training des Modells wurden 75 Epochen gewählt.

<h4>Zusammensetzung der Tracks</h4>
Zur Zusammensetzung der Tracks wird zunächst ein Startsample festgelegt. Daraufhin werden Samples geladen, die potentiell auf das Startsample folgen könnten. Im Anschluss beginnt ein iterativer Prozess: Das Startsample wird mit verfügbaren Samples gematcht, dann bestimmt das Modell für jedes Samplepaar einen Score. Zuletzt wird das Sample mit dem besten Wert an das vorherige Sample angehängt und wird anschließend zum Startsample. Daraufhin beginnt der Prozess mehrmalig von Neuem, bis letztendlich der Export stattfindet. 

<h4>User Interface</h4>
Als User Interface wurde eine simple Eingabe über das Terminal gewählt. Zunächst wird der nutzer gefragt, ob er einen neuen Song erstellen möchte. Daraufhin kann er die gewünschte Songlänge in Sekunden angeben. 

<h4>Ergebnisse</h4>

Mit der Sequence Classification ist ein zufriedenstellendes Ergebnis auf musikalischer Ebene eher unwahrscheinlich. Die Datengrundlage spielt sowohl für das Training als auch für die Songgenerierung eine große Rolle. Für ein annehmbares Ergebnis wäre ein sehr hoher manueller Aufbereitungsaufwand nötig. Um ein besseres Ergebnis zu erreichen müssten die Übergänge zwischen den Samples angepasst werden, auch bei passenden Matches.

<h2>Learnings</h2>
Ein wichtiges Learning aus der Projektarbeit war, dass eine frühere Vertiefung in die Thematik und der Aufbau eines breiten Grundlagenwissens zu einem früheren Zeitpunkt wertvoll gewesen wären. Uns wurde bewusst, dass Tutorials nicht universal übertragbar sind und ohne die Kombination mit einem sicheren Grundlagenwissen nutzlos sind. Außerdem hätte eine bessere Datengrundlage einen positiven Einfluss auf das Endergebnis gehabt. Das Verwenden öffentlicher Datensets sowie vortrainierten Modellen hätte hier Abhilfe schaffen können. 

<h2>Fazit & Ausblick</h2>

<h2>Quellen</h2>

Cheng, Y. (2021). Automatic Music Genre Classification Based on CRNN. Engineering Letters, 29(1).

Frochte, J. (2020). Maschinelles Lernen: Grundlagen und Algorithmen in Python (3., überarbeitete und erweiterte Aufl.). Carl Hanser Verlag GmbH & Co. KG.
Jeong, J. (2021, 7. Dezember). The Most Intuitive and Easiest Guide for Convolutional Neural Network. Medium. Abgerufen am 27. Juni 2022, von https://towardsdatascience.com/the-most-intuitive-and-easiest-guide-for-convolutional-neural-network-3607be47480 

Müller, T. (2021, 6. Juli). Spielarten der Künstlichen Intelligenz: Maschinelles Lernen und Künstliche Neuronale Netze. Fraunhofer IAO – BLOG. Abgerufen am 27. Juni 2022, von https://blog.iao.fraunhofer.de/spielarten-der-kuenstlichen-intelligenz-maschinelles-lernen-und-kuenstliche-neuronale-netze/

Oppermann, A. (2022, 11. Mai). Aktivierungsfunktionen in Neuronalen Netzen: Sigmoid, tanh, ReLU. KI Tutorials. https://artemoppermann.com/de/aktivierungsfunktionen/ 

Srivastava, N. (2014). Dropout: A simple way to prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15.
