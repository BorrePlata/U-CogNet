// Snake 3D Visualization with Three.js
// Visualización en tiempo real del aprendizaje del agente

class Snake3D {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.snakeParts = [];
        this.food = null;
        this.gridSize = 20;
        this.cellSize = 0.5;
        
        this.init();
        this.createGrid();
        this.animate();
        
        // Conectar con el backend Python
        this.connectToBackend();
    }
    
    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111111);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(10, 15, 10);
        this.camera.lookAt(0, 0, 0);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.getElementById('container').appendChild(this.renderer.domElement);
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);
        
        // Controls
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }
    
    createGrid() {
        // Crear grid 3D
        const gridGeometry = new THREE.PlaneGeometry(this.gridSize * this.cellSize, this.gridSize * this.cellSize);
        const gridMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x333333, 
            transparent: true, 
            opacity: 0.3 
        });
        const grid = new THREE.Mesh(gridGeometry, gridMaterial);
        grid.rotation.x = -Math.PI / 2;
        grid.receiveShadow = true;
        this.scene.add(grid);
        
        // Bordes
        const borderMaterial = new THREE.MeshLambertMaterial({ color: 0xff0000 });
        const borderGeometry = new THREE.BoxGeometry(this.gridSize * this.cellSize, 0.1, 0.1);
        
        // Top
        const topBorder = new THREE.Mesh(borderGeometry, borderMaterial);
        topBorder.position.set(0, 0, -this.gridSize * this.cellSize / 2);
        this.scene.add(topBorder);
        
        // Bottom
        const bottomBorder = new THREE.Mesh(borderGeometry, borderMaterial);
        bottomBorder.position.set(0, 0, this.gridSize * this.cellSize / 2);
        this.scene.add(bottomBorder);
        
        // Left
        const leftBorder = new THREE.Mesh(borderGeometry, borderMaterial);
        leftBorder.rotation.z = Math.PI / 2;
        leftBorder.position.set(-this.gridSize * this.cellSize / 2, 0, 0);
        this.scene.add(leftBorder);
        
        // Right
        const rightBorder = new THREE.Mesh(borderGeometry, borderMaterial);
        rightBorder.rotation.z = Math.PI / 2;
        rightBorder.position.set(this.gridSize * this.cellSize / 2, 0, 0);
        this.scene.add(rightBorder);
    }
    
    updateSnake(snakePositions) {
        // Limpiar partes anteriores
        this.snakeParts.forEach(part => this.scene.remove(part));
        this.snakeParts = [];
        
        snakePositions.forEach((pos, index) => {
            const geometry = new THREE.BoxGeometry(this.cellSize * 0.8, this.cellSize * 0.8, this.cellSize * 0.8);
            const material = new THREE.MeshLambertMaterial({ 
                color: index === 0 ? 0x00ff00 : 0x008800 // Cabeza verde, cuerpo verde oscuro
            });
            
            const cube = new THREE.Mesh(geometry, material);
            cube.position.set(
                (pos[0] - this.gridSize / 2) * this.cellSize,
                this.cellSize / 2,
                (pos[1] - this.gridSize / 2) * this.cellSize
            );
            cube.castShadow = true;
            cube.receiveShadow = true;
            
            this.scene.add(cube);
            this.snakeParts.push(cube);
        });
    }
    
    updateFood(foodPos) {
        // Limpiar comida anterior
        if (this.food) {
            this.scene.remove(this.food);
        }
        
        const geometry = new THREE.SphereGeometry(this.cellSize * 0.4);
        const material = new THREE.MeshLambertMaterial({ color: 0xff0000 });
        
        this.food = new THREE.Mesh(geometry, material);
        this.food.position.set(
            (foodPos[0] - this.gridSize / 2) * this.cellSize,
            this.cellSize / 2,
            (foodPos[1] - this.gridSize / 2) * this.cellSize
        );
        this.food.castShadow = true;
        this.food.receiveShadow = true;
        
        this.scene.add(this.food);
    }
    
    updateStats(stats) {
        document.getElementById('episode').textContent = stats.episode || 0;
        document.getElementById('score').textContent = stats.score || 0;
        document.getElementById('steps').textContent = stats.steps || 0;
        document.getElementById('memory').textContent = (stats.memory || 0).toFixed(1);
        document.getElementById('qstates').textContent = stats.qStates || 0;
        document.getElementById('epsilon').textContent = (stats.epsilon || 1).toFixed(2);
        document.getElementById('bestscore').textContent = stats.bestScore || 0;
        document.getElementById('learningrate').textContent = (stats.learningRate || 0).toFixed(2);
    }
    
    connectToBackend() {
        // Simulación: actualizar aleatoriamente para demo
        // En producción, conectar via WebSocket a Python
        setInterval(() => {
            // Simular datos del backend
            const mockStats = {
                episode: Math.floor(Math.random() * 100),
                score: Math.floor(Math.random() * 50),
                steps: Math.floor(Math.random() * 1000),
                memory: 150 + Math.random() * 50,
                qStates: Math.floor(Math.random() * 1000),
                epsilon: Math.random(),
                bestScore: Math.floor(Math.random() * 100),
                learningRate: Math.random() * 0.1
            };
            
            this.updateStats(mockStats);
            
            // Simular snake
            const snakeLength = 5 + Math.floor(Math.random() * 10);
            const snake = [];
            for (let i = 0; i < snakeLength; i++) {
                snake.push([
                    Math.floor(Math.random() * this.gridSize),
                    Math.floor(Math.random() * this.gridSize)
                ]);
            }
            this.updateSnake(snake);
            
            // Simular food
            this.updateFood([
                Math.floor(Math.random() * this.gridSize),
                Math.floor(Math.random() * this.gridSize)
            ]);
        }, 500);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Rotar cámara lentamente
        this.camera.position.x = Math.cos(Date.now() * 0.0005) * 15;
        this.camera.position.z = Math.sin(Date.now() * 0.0005) * 15;
        this.camera.lookAt(0, 0, 0);
        
        this.renderer.render(this.scene, this.camera);
    }
}

// Inicializar cuando la página cargue
window.addEventListener('load', () => {
    new Snake3D();
});