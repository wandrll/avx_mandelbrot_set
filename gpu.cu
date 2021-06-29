
// #include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cassert>
#include <time.h>
#include <cmath>
#include <sys/stat.h>
#include <openssl/sha.h>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>



struct Config{
 
    int window_x;
    int window_y;
    int max_count;
    double R_2;
    double scale;
    double x_center;
    double y_center;
    double dx;
    double dy;
    int last_press;


    void print(){
        printf("window_x = %d\n"
               "window_y = %d\n"
               "R_2 = %lg\n"
               "Max count = %d\n"
               "scale= %lg\n"
               "dx = %lg\n"
               "dy = %lg\n"
               "x center = %lg\n"
               "y center = %lg\n", window_x, window_y, R_2, max_count, scale, dx, dy, x_center, y_center);
               fflush(stdout);
    }

    void init(int winx, int winy, int maxc, double _R_2, double _scale, double _x_center, double _y_center){
        this->window_x = winx;
        this->window_y = winy;
        this->max_count = maxc;
        this->R_2 = _R_2;
        this->scale = _scale;
        this->x_center = _x_center;
        this->y_center = _y_center;
        this->dx = 10 * _scale;
        this->dy = 10 * _scale;
        this->last_press = clock();
    }

    void update(){
        
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)){
            this->x_center += this->dx;
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)){
            this->x_center -= this->dx;
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)){  
            this->y_center += this->dy;
        }
        
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)){    
            this->y_center      -= this->dy;
        }
        
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num0)){
            this->scale *= 1.05; 
            this->dx = 10*this->scale; 
            this->dy = 10*this->scale;
        }
        
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num1)){
            this->scale /= 1.05f; 
            this->dx = 10*this->scale; 
            this->dy = 10*this->scale;
        }
        
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::C)){    
            printf("x = %e\n y = %e\n scale = %e\n", this->x_center, this->y_center, this->scale);
        }
    }


};



__device__ const int pallete_size = 16;
__device__ const unsigned colors[pallete_size + 1] = {
        66    +   256 * 30    +   256 * 256 * 15    +   256 * 256 * 256 * 255,
        25    +   256 * 7     +   256 * 256 * 26    +   256 * 256 * 256 * 255,
        9     +   256 * 1     +   256 * 256 * 47    +   256 * 256 * 256 * 255,
        4     +   256 * 4     +   256 * 256 * 73    +   256 * 256 * 256 * 255,
        0     +   256 * 7     +   256 * 256 * 100   +   256 * 256 * 256 * 255,
        12    +   256 * 44    +   256 * 256 * 138   +   256 * 256 * 256 * 255,
        24    +   256 * 82    +   256 * 256 * 177   +   256 * 256 * 256 * 255,
        57    +   256 * 125   +   256 * 256 * 209   +   256 * 256 * 256 * 255,
        134   +   256 * 181   +   256 * 256 * 229   +   256 * 256 * 256 * 255,
        211   +   256 * 236   +   256 * 256 * 248   +   256 * 256 * 256 * 255,
        241   +   256 * 233   +   256 * 256 * 191   +   256 * 256 * 256 * 255,
        248   +   256 * 201   +   256 * 256 * 95    +   256 * 256 * 256 * 255,
        255   +   256 * 170   +   256 * 256 * 0     +   256 * 256 * 256 * 255,
        204   +   256 * 128   +   256 * 256 * 0     +   256 * 256 * 256 * 255,
        153   +   256 * 87    +   256 * 256 * 0     +   256 * 256 * 256 * 255,
        106   +   256 * 52    +   256 * 256 * 3     +   256 * 256 * 256 * 255,
         0    +       0       +           0         +   256 * 256 * 256 * 255,
};

__device__ unsigned get_color(int n, Config* cfg){
    if (n < cfg->max_count) {
        return colors[n%pallete_size];
    }else{
        return colors[pallete_size];
    }
}

__global__ void calculate_double(Config* cfg, unsigned* result){
    int curr = (blockIdx.x << 10) + threadIdx.x;
    int indexy = curr/cfg->window_x;
    int indexx = curr%cfg->window_x;
    
    double y0 =  ((double)indexy - (cfg->window_y >> 1))*cfg->scale + cfg->y_center;
    double x0 =  ((double)indexx - (cfg->window_x >> 1))*cfg->scale + cfg->x_center;
    double X = x0, 
           Y = y0;

    
    int N = 0;
    
    for (; N < cfg->max_count; N++){
      double x2 = X*X,
            y2 = Y*Y,
            xy = X*Y;
    
      double r2 = x2 + y2;
         
      if (r2 >= cfg->R_2) break;
          
      X = x2 - y2 + x0,
      Y = xy + xy + y0;
    }
    result[indexy * cfg->window_x + indexx]= get_color(N, cfg);

}

__global__ void calculate_float(Config* cfg, unsigned* result){
    int curr = (blockIdx.x << 10) + threadIdx.x;
    int indexy = curr/cfg->window_x;
    int indexx = curr%cfg->window_x;
    
    float y0 =  ((float)indexy - (cfg->window_y >> 1))*cfg->scale + cfg->y_center;
    float x0 =  ((float)indexx - (cfg->window_x >> 1))*cfg->scale + cfg->x_center;
    float X = x0, 
           Y = y0;

    
    int N = 0;
    
    for (; N < cfg->max_count; N++){
      float x2 = X*X,
            y2 = Y*Y,
            xy = X*Y;
    
      float r2 = x2 + y2;
         
      if (r2 >= cfg->R_2) break;
          
      X = x2 - y2 + x0,
      Y = xy + xy + y0;
    }
    result[indexy * cfg->window_x + indexx]= get_color(N, cfg);

}



void printf_fps(){
    static clock_t lastTime = clock();
    clock_t currentTime = clock();
    double fps = 1.f / ((double)(currentTime - lastTime)/ CLOCKS_PER_SEC);
    lastTime = currentTime;
    printf("%g\r", fps);
    fflush(stdout);
}


void render(Config* cfg){
    int prev_iterations = cfg->max_count;
    cfg->max_count = 8192;

    unsigned* field = NULL;     
    char* name = (char*)calloc(40, sizeof(int));
    sprintf(name, "images/%d.jpeg", clock());

    int pixels_count = cfg->window_x * cfg->window_y;
    int block_count = pixels_count / 1024 + 1;
    cudaMallocManaged(&field, (block_count + 1) * 1024 * sizeof(unsigned));

    calculate_double<<<block_count, 1024>>>(cfg, field);
    cudaDeviceSynchronize();

    sf::Image img;
    img.create(cfg->window_x, cfg->window_y, (sf::Uint8*)field);
    img.saveToFile(name);

    cudaFree(field);
    free(name);
    cfg->max_count = prev_iterations;
}


void draw_mandelbrott(Config* cfg){

    sf::RenderWindow window(sf::VideoMode(cfg->window_x, cfg->window_y), "BLABLABLA");
    window.setPosition(sf::Vector2i(430, 200));

    sf::Image image;
    image.create(cfg->window_x, cfg->window_y, sf::Color::Cyan);

    sf::Texture texture;
    texture.loadFromImage(image);


    sf::Sprite sprite;
    sprite.setTexture(texture);


    unsigned* field = NULL;     

    int pixels_count = cfg->window_x * cfg->window_y;
    int block_count = pixels_count / 1024 + 1;

    cudaMallocManaged(&field, (block_count + 1) * 1024 * sizeof(unsigned));


    while (window.isOpen()){
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))     break;
        if ((clock()-cfg->last_press)/CLOCKS_PER_SEC >= 2 && sf::Keyboard::isKeyPressed(sf::Keyboard::P)){
            render(cfg);
            cfg->last_press = clock();
        }    
        cfg->update();

        calculate_float<<<block_count, 1024>>>(cfg, field);
        cudaDeviceSynchronize();

       
        

        printf_fps();
        // printf("x = %4e y = %4e %4e\r",cfg->x_center, cfg->y_center, cfg->scale);
        // cfg->print();
        fflush(stdout);
        texture.update((sf::Uint8*)field, cfg->window_x, cfg->window_y, 0, 0);

        sprite.setTexture(texture);

        window.draw(sprite);
        window.display();
    }
    
    cudaFree(field);
}



int main(){

   
    Config* cfg = NULL;
    cudaMallocManaged(&cfg, sizeof(Config));
    cfg->init(1600, 900, 1024, 4,  1/120.f,  0,  0);
    
    
    draw_mandelbrott(cfg);
    
    cfg->init(7680, 4800, 8192, 4, 5.802513e-06        ,  -7.374954e-01,  -2.084319e-01);

    // render(cfg);


    cudaFree(cfg);

    

  return 0;
}