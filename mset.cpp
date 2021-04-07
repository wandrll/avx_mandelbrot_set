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
#include <immintrin.h>

enum Video_Mode{
    SSE_MODE,
    NON_SSE_MODE
};

const int pallete_size = 16;

const unsigned colors[pallete_size + 1] = {
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


struct Config{

    int window_x;
    int window_y;
    const double R_2 = 4;
    const int max_count = 255;
    double scale;
    double dx;
    double dy;
    double x_center;
    double y_center;
    bool test;
    Video_Mode mode;
    __m256d __R_2;
    int previous_pressing;

    void init(int _window_x, int _window_y, double _scale, double _dx, double _dy, double _x_center, double _y_center){
        this->window_x = _window_x;
        this->window_y = _window_y;
        this->scale = _scale;
        this->dx = _dx;
        this->dy = _dy;
        this->x_center = _x_center;
        this->y_center = _y_center;
        this->mode = NON_SSE_MODE;
        this->previous_pressing = 0;
        this->__R_2 = _mm256_set1_pd(this->R_2);
    }

    void update(){
        
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q)){
            this->mode = NON_SSE_MODE;
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)){
            this->mode = SSE_MODE;
        }





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
            this->y_center -= this->dy;
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
        
    }


};


unsigned get_color(int n, Config* cfg){
    if (n < cfg->max_count) {
        return colors[n%pallete_size];
    }else{
        return colors[pallete_size];
    }
}

__m256i quad_calcule(Config* cfg, double x0, double y0){
    __m256d __scale  = _mm256_set1_pd(cfg->scale);
    __m256d __offset = _mm256_set_pd (3, 2, 1, 0);
    __m256d __X0     = _mm256_set1_pd(x0);
    __X0 = _mm256_add_pd(__X0, _mm256_mul_pd(__scale,  __offset));      

    __m256d __Y0      = _mm256_set1_pd(y0);

    __m256d __curr_X = __X0;
    __m256d __curr_Y = __Y0;

    __m256i __N     = _mm256_set1_epi64x(0);


    for (int n = 0; n < cfg->max_count; n++){
        __m256d __sqrX = _mm256_mul_pd(__curr_X, __curr_X);
        __m256d __sqrY = _mm256_mul_pd(__curr_Y, __curr_Y);
        __m256d __XY = _mm256_mul_pd(__curr_X, __curr_Y);

        __m256d __r2 = _mm256_add_pd(__sqrX, __sqrY);    
        __m256d __cmp     = _mm256_cmp_pd(__r2, cfg->__R_2, 2);

        int mask = _mm256_movemask_pd(__cmp);
        if (!mask){
            break;
        }
        __N = _mm256_sub_epi32(__N, _mm256_castpd_si256(__cmp));
        __curr_X = _mm256_sub_pd(_mm256_add_pd(__sqrX,__X0), __sqrY);
        __curr_Y = _mm256_add_pd(__XY, _mm256_add_pd(__XY, __Y0));
    }
    return __N;
}

void calculate_field_sse(Config* cfg, unsigned* field){
    for (int iy = 0; iy < cfg->window_y; iy++) {

        double y0 =  ((double)iy - cfg->window_y/2)*cfg->scale + cfg->y_center;
        double x0 =  (           - cfg->window_x/2)*cfg->scale + cfg->x_center;
        
        for (int ix = 0; ix < cfg->window_x; ix+=4, x0 += 4*cfg->scale){        
            __m256i __N = quad_calcule(cfg, x0, y0);
            for (int i = 0; i < 4; i++){
                field[iy*(cfg->window_x) + ix + i] = get_color(__N[i], cfg);
            } 
        }

    }
}




void calculate_field_nonsse(Config* cfg, unsigned* field){ 
    for (int iy = 0; iy < cfg->window_y; iy++) {

        double y0 =  ((double)iy - cfg->window_y/2)*cfg->scale + cfg->y_center;
        double x0 =  (           - cfg->window_x/2)*cfg->scale + cfg->x_center;
        for (int ix = 0; ix < cfg->window_x; ix++, x0 += cfg->scale){       
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

            field[iy*(cfg->window_x) + ix] = get_color(N, cfg);

        }
    }
}


void printf_fps(){
    static clock_t lastTime = clock();
    clock_t currentTime = clock();
    double fps = 1.f / ((double)(currentTime - lastTime)/ CLOCKS_PER_SEC);
    lastTime = currentTime;
    printf("%g\r", fps);
    fflush(stdout);
}


void draw_mandelbrott(Config* cfg){


    sf::RenderWindow window(sf::VideoMode(cfg->window_x, cfg->window_y), "Blablabla");
    window.setPosition(sf::Vector2i(430, 200));

    sf::Image image = {};
    image.create(cfg->window_x, cfg->window_y, sf::Color::Cyan);
    
    sf::Texture texture = {};
    texture.loadFromImage(image);


    sf::Sprite sprite = {};
    sprite.setTexture(texture);

    sf::Uint8* field = (sf::Uint8*)calloc(cfg->window_x * cfg->window_y, sizeof(unsigned));  


    while (window.isOpen()){
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) break;
        
        cfg->update();
        switch(cfg->mode){
            case SSE_MODE:{

                calculate_field_sse(cfg, (unsigned*)field);
             
                break;
            }
            case NON_SSE_MODE:{

                calculate_field_nonsse(cfg, (unsigned*)field);

                break;
            }
            default:{
                printf("Error, weird calculation mode\n");
                return;
            }
        }
        
        printf_fps();
        
        texture.update(field, cfg->window_x, cfg->window_y, 0, 0);
        sprite.setTexture(texture);

        window.draw(sprite);
        window.display();
    }
    free(field); 

}



int main(){

    Config cfg = {};
    cfg.init(4*200, 4*150, 1/120.f, 1/12.f, 1/12.f, 0.01, 0.005);
    draw_mandelbrott(&cfg);

}