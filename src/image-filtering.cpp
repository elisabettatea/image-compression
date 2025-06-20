#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdlib> // Per std::rand()
#include <unsupported/Eigen/SparseExtra>
#include <fstream>
#include <vector>
#include <cstdlib> // Per std::system
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
typedef Eigen::Triplet<double> T;

int main(int argc, char* argv[]) {
    // Check if the image path is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char* input_image_path = argv[1];

    // Load the grayscale image
    int width, height, channels;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);

    if (!image_data) {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channel(s)." << std::endl;
    std::cout << "Matrix: number of rows = " << width << "; number of columns = " << height << ";" << std::endl;

    // Create an Eigen matrix from the image data
    MatrixXd image_matrix(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            image_matrix(i, j) = static_cast<double>(image_data[i * width + j]);
        }
    }

    stbi_image_free(image_data); // Free the loaded image data


    MatrixXd noise_matrix(height, width);

    // Aggiungi il rumore all'immagine
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int noise = std::rand() % 101 - 50; // Rumore casuale tra -50 e 50
            int newValue = static_cast<int>(image_matrix(i, j)) + noise; // Somma il rumore al pixel
            newValue = std::max(0, std::min(255, newValue)); // Limita il valore tra 0 e 255
            noise_matrix(i, j) = static_cast<double>(newValue); // Assegna il valore alla matrice del rumore
        }
    }

    // Converti la matrice del rumore in formato unsigned char (0-255) per salvarla come immagine
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> output_image(height, width);
    output_image = noise_matrix.cast<unsigned char>();

    // Salva l'immagine di output con il rumore
    const std::string output_image_path = "output_image_noise.png";
    if (stbi_write_png(output_image_path.c_str(), width, height, 1, output_image.data(), width) == 0) {
        std::cerr << "Error: Could not save output image" << std::endl;
        return 1;
    }

    std::cout << "Output image with noise saved to " << output_image_path << std::endl;


    // Reshape original image and noisy image as vectors 
    VectorXd v(height*width);
    VectorXd w(height*width);

    // Verifica delle dimensioni dei vettori
    std::cout << "Dimensione del vettore v: " << v.size() << std::endl;
    std::cout << "Dimensione del vettore w: " << w.size() << std::endl;

    for (int i = 0; i < height; ++i){
      for (int j=0; j < width; ++j){
          v(i*width+j)=image_matrix(i, j);
          w(i*width+j)=noise_matrix(i ,j);
      }
    }

    // Calcolo della norma euclidea del vettore originale v
    double norm_v = v.norm();
    std::cout << "Norma euclidea di v: " << norm_v << std::endl;

    // Create the convolution kernel (3x3 smoothing filter)
    MatrixXd Hav2 = (1.0 / 9) * MatrixXd::Ones(3, 3);
    SparseMatrix<double> A1(height * width, height * width); // Sparse matrix for convolution
    std::vector<T> tripletList;
    tripletList.reserve(9*height*width);
    int index;

    
    // Build the sparse matrix A1
    for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
        index = i * width + j;
        
        // Centro del kernel
        tripletList.push_back(T(index, index, Hav2(1,1)));
        
        // Sopra
        if (i > 0) {
            tripletList.push_back(T(index, (i - 1) * width + j, Hav2(0, 1)));
            
            // Sopra sinistra
            if (j > 0) {
                tripletList.push_back(T(index, (i - 1) * width + j - 1, Hav2(0, 0)));
            }
            // Sopra destra
            if (j < width - 1) {
                tripletList.push_back(T(index, (i - 1) * width + j + 1, Hav2(0, 2)));
            }
        }
        
        // Sinistra
        if (j > 0) {
            tripletList.push_back(T(index, i * width + j - 1, Hav2(1, 0)));
        }
        
        // Destra
        if (j < width - 1) {
            tripletList.push_back(T(index, i * width + j + 1, Hav2(1, 2)));
        }
        
        // Sotto
        if (i < height - 1) {
            tripletList.push_back(T(index, (i + 1) * width + j, Hav2(2, 1)));
            
            // Sotto sinistra
            if (j > 0) {
                tripletList.push_back(T(index, (i + 1) * width + j - 1, Hav2(2, 0)));
            }
            // Sotto destra
            if (j < width - 1) {
                tripletList.push_back(T(index, (i + 1) * width + j + 1, Hav2(2, 2)));
            }
        }
    }
}

  A1.setFromTriplets(tripletList.begin(), tripletList.end());

 // stampa angolo A1
  std::cout << "A1 head "<<std::endl << A1.topLeftCorner(20,20) << std::endl;

 // stampa elementi non nulli A1
 std::cout << "A1 non zero elements: " << A1.nonZeros() << std::endl;

 // Convoluzione tra matrice sparsa A1 e il vettore w 
 VectorXd aw=A1*w;

 // crea una matrice smooth con i risultati della convoluzione applicata all'immagine
 MatrixXd smooth(height, width);

    for (int i = 0; i < height; ++i){
        for (int j=0; j < width; ++j){
            index = i*width + j;
            if(aw(index)<0){
            aw(index)=0;
            }
            if(aw(index)>255){
            aw(index)=255;
            }
            smooth(i,j)=aw(index);
        }
    }

 Matrix<unsigned char, Dynamic, Dynamic, RowMajor> smooth_image(height, width);
  smooth_image = smooth.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val);
  });
 
  //Save the smoothed image using stb_image_write
  const std:: string output_smooth_image_path = "output_smooth.png";
  if (stbi_write_png(output_smooth_image_path.c_str(), width, height, 1, smooth_image.data(), width) == 0) {
    std::cerr << "Error: Could not save smoothed image" << std::endl;

    return 1;
  }

  std::cout << "Smoothed image saved to " << output_smooth_image_path << std::endl;

 // rifaccio stessa cosa per il filtro sharpening
 // sharpening kernel 3x3
 Eigen::SparseMatrix<double> A2(height*width, height*width);
 std::vector<T> tripletList2;
 tripletList2.reserve(5*height*width);

 // build matrice A2 sparse
 
  for (int i = 0; i < height; ++i){ //punto che prendo in considerazione
    for (int j=0; j < width; ++j){
      index = i*width+j;
      tripletList2.push_back(T(index, index, 9.0));
 //!row non cambia mai in A, mi sposto su column
      if(i>0){
      tripletList2.push_back(T(index, (i-1)*width +j, -3.0));
      }

      if(j>0){
        tripletList2.push_back(T(index, i*width+j-1, -1.0));
      }
      if(j< width-1){
        tripletList2.push_back(T(index, i*width+j+1, -3.0));
      }
      if( i< height-1){
        tripletList2.push_back(T(index, (i+1)*width +j, -1.0));
      }
    }
  }


  A2.setFromTriplets(tripletList2.begin(), tripletList2.end());

// stampa angolo A2
std::cout << "A2 head "<<std::endl << A2.topLeftCorner(20,20) << std::endl;

  //stampo elementi non nulli di A2

   std::cout << "A2 non zero elements: " << A2.nonZeros() << std::endl;

 // Check symmetria di A2
 SparseMatrix<double> B = SparseMatrix<double>(A2.transpose()) - A2; 
 std::cout << "Norm of skew-symmetric part: " << B.norm() << std::endl;
 std::cout <<  "If Norm == 0, matrix is symmetric " << B.norm() << std::endl;

 // Convoluzione tra matrice sparsa A1 e il vettore w 
 VectorXd av=A2*v;

// crea una matrice sharp con i risultati della convoluzione applicata all'immagine
 MatrixXd sharp(height, width);

    for (int i = 0; i < height; ++i){
        for (int j=0; j < width; ++j){
            index = i*width + j;
            if(av(index)<0){
            av(index)=0;
            }
            if(av(index)>255){
            av(index)=255;
            }
            sharp(i,j)=av(index);
        }
    }
 
 Matrix<unsigned char, Dynamic, Dynamic, RowMajor> sharp_image(height, width);
  sharp_image = sharp.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val);
  });


 //Save the perturbed image using stb_image_write
  const std:: string output_sharp_image_path = "output_sharp.png";
  if (stbi_write_png(output_sharp_image_path.c_str(), width, height, 1, sharp_image.data(), width) == 0) {
    std::cerr << "Error: Could not save sharp image" << std::endl;

    return 1;
  }

  std::cout << "Sharp image saved to " << output_sharp_image_path << std::endl;

  // Exporting A2 to Matrix Market format
    std::string matrixFileOut("./A2.mtx");
    if (!Eigen::saveMarket(A2, matrixFileOut)) {
        std::cerr << "Error saving matrix A2 to " << matrixFileOut << std::endl;
        return 1;
    }
    std::cout << "Matrix A2 saved to " << matrixFileOut << std::endl;

    // Exporting w to Matrix Market format
    std::string vectorFileOut("./w.mtx");
    if (!Eigen::saveMarket(w, vectorFileOut)) {
        std::cerr << "Error saving vector w to " << vectorFileOut << std::endl;
        return 1;
    }

    std::cout << "Vector w saved to " << vectorFileOut << std::endl;

   // cambio header del vettore a "%%MatrixMarket vector coordinate real general" per farlo leggere correttamente a lis

    // Legge il contenuto del file originale
    std::string fileName = "w.mtx"; // Nome del file da modificare

    // Apri il file di input
    std::ifstream infile(fileName);
    if (!infile.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file " << fileName << std::endl;
        return 1;
    }
  // Leggi il contenuto del file originale, ignorando la prima riga
    std::string line;
    std::string content;
    
    // Ignora la prima riga (la vecchia intestazione)
    std::getline(infile, line); 

    // Leggi il resto del file
    while (std::getline(infile, line)) {
        content += line + "\n"; // Aggiungi ogni riga al contenuto
    }

    // Chiudi il file di input
    infile.close();

    // Apri il file per la scrittura (sovrascrivendo il file originale)
    std::ofstream outfile(fileName);
    if (!outfile.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file " << fileName << std::endl;
        return 1;
    }

    // Scrivi la nuova intestazione
    outfile << "%%MatrixMarket vector coordinate real general\n";

    // Scrivi il contenuto originale nel file con la nuova intestazione
    // La dimensione del vettore viene letta dalla prima riga
    // int num_rows = 87296; // Numero di righe dal tuo esempio
    // outfile << num_rows << "\n"; // Scrivi solo il numero di righe

    outfile << content; // Scrivi il contenuto rimanente

    // Chiudi il file di output
    outfile.close();

    std::cout << "File " << fileName << " modificato con la nuova intestazione." << std::endl;
    
    // stampo head di w
    // std::cout << "Primi 10 valori di w:" << std::endl;
    // std::cout << w.head(10) << std::endl; // Stampa i primi 10 valori di w

    
    //salvo in sol.txt la soluzione del sistema A2x=w con tol=1e-9

    std::cout << "Eseguo test1 con sistema: A2 * x = w"  << std::endl;

    // Comando da eseguire
     const char* command = "mpirun -n 4 ./test1 A2.mtx w.mtx sol.txt hist.txt -i jacobi -tol 1.0e-9";

   // Esegui il comando
    int result = std::system(command);

    // Controlla il risultato dell'esecuzione
    if (result == 0) {
        std::cout << "Test1 eseguito con successo." << std::endl;
    } else {
        std::cerr << "Errore durante l'esecuzione del test1." << std::endl;
    }
    

    //importo la soluzione sol.txt in vettore x di eigen e lo converto a .png, poi lo esporto

   // Dimensione del vettore x
    int n = height * width; // Assicurati che questa sia la dimensione corretta

    // Crea un vettore per contenere i dati letti dal file
    VectorXd x(n);


    // Se hai già dichiarato infile e line in precedenza
// non è necessario ridefinirli qui

// Usa infile per aprire "sol.txt" 
std::ifstream infileSol("sol.txt");
if (!infileSol.is_open()) {
    std::cerr << "Errore: Impossibile aprire il file sol.txt" << std::endl;
    return 1;
}

// Se hai già dichiarato 'line', non dichiararla di nuovo
// std::string line; // Questa riga è già stata dichiarata in precedenza

// Ignora la prima riga di intestazione
std::getline(infileSol, line); // Prima riga: %%MatrixMarket vector coordinate real general

// Leggi la seconda riga che contiene il numero di elementi
std::getline(infileSol, line); // Seconda riga: numero di elementi (87296)

int expected_count;
try {
    expected_count = std::stoi(line); // Converti la riga in un numero intero
} catch (const std::invalid_argument&) {
    std::cerr << "Errore: Impossibile convertire la riga in un numero. Contenuto: '" << line << "'" << std::endl;
    return 1;
}

// Controlla se il numero di valori letti è corretto
if (expected_count != n) {
    std::cerr << "Errore: Numero atteso di elementi (" << n << ") non corrisponde a quanto dichiarato nel file (" << expected_count << ")" << std::endl;
    return 1;
}

// Variabile per contare i valori letti
int count = 0;

// Leggi i dati nel vettore
while (count < n && infileSol >> line) {
    // Ignora il primo valore della riga (indice)
    int index;
    infileSol >> index; // Leggi l'indice
    double value;
    infileSol >> value; // Leggi il valore
    x(count) = value; // Assegna il valore al vettore
    count++;
}

infileSol.close();

std::cout << "Numero di valori letti da sol.txt: " << count << std::endl;
std::cout << "Primi 5 valori: " << x.head(5).transpose() << std::endl; // Stampa i primi 5 valori letti

// Controlla se il numero di valori letti corrisponde a n
if (count != n) {
    std::cerr << "Errore: Numero di valori letti da sol.txt (" << count << ") non corrisponde a n (" << n << ")." << std::endl;
    return 1;
}

// Crea una matrice 2D da x (valori da 0 a 1)
  
  MatrixXd imgsol9(height, width); // Crea la matrice di output

  
  // Popola la matrice con i dati da x
  for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
          imgsol9(i, j) = x(i * width + j); // Mappa i dati da x alla matrice
      }
  }

  

   // Stampa l'angolo superiore sinistro di imgsol9 (20x20) utilizzando cicli for
  std::cout << "imgsol9 head:" << std::endl;
  for (int i = 0; i < std::min(3, height); ++i) { // Itera sulle righe (fino a 20 o fino alla fine della matrice)
      for (int j = 0; j < std::min(3, width); ++j) { // Itera sulle colonne (fino a 20 o fino alla fine della matrice)
          std::cout << imgsol9(i, j) << " "; // Stampa il valore nella posizione (i, j)
      }
      std::cout << std::endl; // Vai a capo dopo ogni riga
  }

  
    // Normalizza i valori a 0-255 e converte in unsigned char
    Matrix<unsigned char, Dynamic, Dynamic> imgsol9_normalized(height, width);
    imgsol9_normalized = imgsol9.unaryExpr([](double val) {
      return static_cast<unsigned char>(std::min(std::max(val * 255.0, 0.0), 255.0));
  });


    // Salva l'immagine come PNG
    const std::string output_imgsol9_image_path = "x_image.png";
    if (stbi_write_png(output_imgsol9_image_path.c_str(), width, height, 1, imgsol9_normalized.data(), width) == 0) {
        std::cerr << "Errore: Impossibile salvare l'immagine" << std::endl;
        return 1;
    }

    std::cout << "Immagine salvata come " << output_imgsol9_image_path << std::endl;


  // edge 
  // da qui in giù l'esecuzione del codice viene interrotta (killed), ma la porzione di codice gira separatamente 
  // ci ha fornito le immagini in output edge_image e y_image

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> edge_image(height,width);

  //rifaccio stessa cosa per laplacian edge detection
  // costruisco matrice A3
  MatrixXd Hlap = MatrixXd::Zero(n,n);

  Hlap(0,1)=-1.0;
  Hlap(1,0)=-1.0;
  Hlap(1,1)=4.0;
  Hlap(1,2)=-1.0;
  Hlap(2,1)=-1.0;

  SparseMatrix<double> A3(height*width, height*width);

  std::vector<T> tripletList3;
  tripletList3.reserve(height*width*5);

  for (int i = 0; i < height; ++i)  //punto che prendo in considerazione
  { 
    for (int j=0; j < width; ++j)
    {
      int index = i*width+j;
      tripletList3.push_back(T(index, index, Hlap(1,1)));
      //!row non cambia mai in A, mi sposto su column
      if(i>0)
      {
        tripletList3.push_back(T(index, (i-1)*width +j, Hlap(0,1)));
      }

      if(j>0)
      {
        tripletList3.push_back(T(index, i*width+j-1, Hlap(1,0)));
      }
      if(j< width-1)
      {
        tripletList3.push_back(T(index, i*width+j+1, Hlap(1,2)));
      }
      if( i< height-1)
      {
        tripletList3.push_back(T(index, (i+1)*width +j, Hlap(2,1)));
      }
    }
  }
  
  A3.setFromTriplets(tripletList3.begin(), tripletList3.end());

 //stampo testa matrice A3
  std::cout << "A3 head "<<std::endl << A3.topLeftCorner(20,20) << std::endl;

 //stampo numero elementi non nulli in A3
  std::cout << "A3 non zero elements "<<std::endl << A3.nonZeros() << std::endl;
  
  //check su simmetria di A3
  B = SparseMatrix<double>(A3.transpose()) - A3;  
  std::cout << "Norm of skew-symmetric part: " << B.norm() << std::endl;
  std::cout <<  "If Norm == 0, matrix is symmetric " << B.norm() << std::endl;

// Convoluzione tra matrice sparsa A3 e il vettore 3
VectorXd filt(height*width); 
 filt=A3*v;

 // crea una matrice edge con i risultati della convoluzione applicata all'immagine
 MatrixXd edge(height, width);

  for (int i=0; i<height; i++)
  {
    for (int j=0; j<width; j++)
    {
      int index = i*width+j;
      edge(i,j)=filt(index);
      if (edge (i,j)>255.0)
      {
        edge(i,j)=255.0;  
      }
      if (edge(i,j)<0.0)
      {
        edge(i,j)=0.0;
      }
    }
  }

//use Eigen's unaryExpr to map the greyscale values (0.0 to 1.0) to 0 to 255
  edge_image = edge.unaryExpr([](double val) -> unsigned char
  {
    return static_cast<unsigned char>(val);
  });

  //Save the image using stb_image_write
  const std::string output_image_path4 = "edge_image.png";
  if (stbi_write_png(output_image_path4.c_str(),width,height,1,edge_image.data(),width)==0)
  {
    std::cout << "Error: Could not save greyscale image" << std::endl;

    return 1;
  }

  std::cout << "Image saved to " << output_image_path4 << std::endl; 

//creo matrice identità 
SparseMatrix<double> id(height*width, height*width);
for (int i = 0; i < height*width; i++) {
    id.coeffRef(i, i) = 1.0;
}

// Somma tra identità e A3 (matrice simmetrica)
B = id + A3;  


// Definizione del vettore risultato
Eigen::VectorXd y(B.rows());  // Corretta inizializzazione di 'y'

// Settaggio dei parametri per il solver
double tol = 1.e-10;           // Tolleranza di convergenza
int maxit = 1000;              // Numero massimo di iterazioni

// Creazione del precondizionatore diagonale (non necessario in questo contesto se usi solo CG)
Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;   // Solver CG
cg.setMaxIterations(maxit);     // Imposta le iterazioni massime
cg.setTolerance(tol);           // Imposta la tolleranza
cg.compute(B);                  // Precomputa i fattori necessari
y = cg.solve(w);                // Risolve il sistema B*y = w

// Visualizzazione dei risultati
std::cout << "Eigen native CG" << std::endl;
std::cout << "#iterations:     " << cg.iterations() << std::endl;
std::cout << "relative residual: " << cg.error() << std::endl;


//std::cout << "solution : y = " << y << std::endl;

//creo matrice per y (mat-y)
Matrix<unsigned char, Dynamic, Dynamic, RowMajor> y_image(height,width);

MatrixXd maty(height, width);

//inserisco valori in maty di y
for (int i=0; i<height; i++)
{
  for (int j=0; j<width; j++)
  {
    int index = i*width+j;
    maty(i,j)=y(index);
    if (maty(i,j)>255.0)
    {
      maty(i,j)=255.0;  
    }
    if (maty(i,j)<0.0)
    {
      maty(i,j)=0.0;
    }
  }
}

//use Eigen's unaryExpr to map the greyscale values (0.0 to 1.0) to 0 to 255
y_image = maty.unaryExpr([](double val) -> unsigned char
{
  return static_cast<unsigned char>(val);
});

//Save the image using stb_image_write
const std::string output_image_path5 = "y_image.png";
if (stbi_write_png(output_image_path5.c_str(),width,height,1,y_image.data(),width)==0)
{
  std::cout << "Error: Could not save greyscale image" << std::endl;

  return 1;
}

std::cout << "Image saved to " << output_image_path5 << std::endl; 



    return 0;
}
