import { useState, useEffect, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Shield, Zap, Target, Coins, Heart, Clock, Bot, User } from 'lucide-react'
import './App.css'

// ゲーム定数（バランス調整済み）
const GAME_CONFIG = {
  CANVAS_WIDTH: 800,
  CANVAS_HEIGHT: 600,
  TOWER_COST: 50,
  TOWER_DAMAGE: 60, // ダメージを大幅増加
  TOWER_RANGE: 150, // 射程を大幅拡大
  ENEMY_HEALTH: 80, // 敵の体力を減少
  ENEMY_SPEED: 0.7, // 敵の速度を減少
  ENEMY_REWARD: 30, // 報酬を3倍に増加
  WAVE_SIZE: 3, // ウェーブサイズを減少
  INITIAL_MONEY: 250, // 初期資金を増加
  INITIAL_HEALTH: 100
}

// パス定義（敵の移動経路）
const PATH = [
  { x: 0, y: 300 },
  { x: 200, y: 300 },
  { x: 200, y: 150 },
  { x: 400, y: 150 },
  { x: 400, y: 450 },
  { x: 600, y: 450 },
  { x: 600, y: 300 },
  { x: 800, y: 300 }
]

// ゲーム状態クラス
class GameState {
  constructor() {
    this.reset()
  }

  reset() {
    this.money = GAME_CONFIG.INITIAL_MONEY
    this.health = GAME_CONFIG.INITIAL_HEALTH
    this.wave = 1
    this.score = 0
    this.towers = []
    this.enemies = []
    this.projectiles = []
    this.gameTime = 0
    this.waveTimer = 0
    this.isGameRunning = false
    this.gameOver = false
  }

  addTower(x, y) {
    if (this.money >= GAME_CONFIG.TOWER_COST && this.canPlaceTower(x, y)) {
      this.towers.push({
        x: x,
        y: y,
        damage: GAME_CONFIG.TOWER_DAMAGE,
        range: GAME_CONFIG.TOWER_RANGE,
        lastShot: 0,
        kills: 0
      })
      this.money -= GAME_CONFIG.TOWER_COST
      return true
    }
    return false
  }

  canPlaceTower(x, y) {
    // パス上には配置できない
    for (let i = 0; i < PATH.length - 1; i++) {
      const p1 = PATH[i]
      const p2 = PATH[i + 1]
      const dist = this.distanceToLine(x, y, p1.x, p1.y, p2.x, p2.y)
      if (dist < 30) return false
    }
    
    // 他のタワーと重複しない
    for (const tower of this.towers) {
      if (Math.hypot(x - tower.x, y - tower.y) < 40) return false
    }
    
    return true
  }

  distanceToLine(px, py, x1, y1, x2, y2) {
    const A = px - x1
    const B = py - y1
    const C = x2 - x1
    const D = y2 - y1
    
    const dot = A * C + B * D
    const lenSq = C * C + D * D
    let param = -1
    if (lenSq !== 0) param = dot / lenSq
    
    let xx, yy
    if (param < 0) {
      xx = x1
      yy = y1
    } else if (param > 1) {
      xx = x2
      yy = y2
    } else {
      xx = x1 + param * C
      yy = y1 + param * D
    }
    
    const dx = px - xx
    const dy = py - yy
    return Math.sqrt(dx * dx + dy * dy)
  }

  spawnWave() {
    for (let i = 0; i < GAME_CONFIG.WAVE_SIZE; i++) {
      setTimeout(() => {
        this.enemies.push({
          x: PATH[0].x,
          y: PATH[0].y,
          health: GAME_CONFIG.ENEMY_HEALTH * (1 + this.wave * 0.2),
          maxHealth: GAME_CONFIG.ENEMY_HEALTH * (1 + this.wave * 0.2),
          speed: GAME_CONFIG.ENEMY_SPEED * (1 + this.wave * 0.1),
          pathIndex: 0,
          progress: 0
        })
      }, i * 1000)
    }
  }

  update() {
    if (!this.isGameRunning || this.gameOver) return

    this.gameTime += 1/60
    this.waveTimer += 1/60

    // 新しいウェーブの生成
    if (this.enemies.length === 0 && this.waveTimer > 3) {
      this.wave++
      this.spawnWave()
      this.waveTimer = 0
    }

    // 敵の移動
    this.enemies.forEach((enemy, enemyIndex) => {
      if (enemy.pathIndex < PATH.length - 1) {
        const current = PATH[enemy.pathIndex]
        const next = PATH[enemy.pathIndex + 1]
        const dx = next.x - current.x
        const dy = next.y - current.y
        const distance = Math.hypot(dx, dy)
        
        enemy.progress += enemy.speed / distance
        
        if (enemy.progress >= 1) {
          enemy.pathIndex++
          enemy.progress = 0
        }
        
        if (enemy.pathIndex < PATH.length - 1) {
          const currentPath = PATH[enemy.pathIndex]
          const nextPath = PATH[enemy.pathIndex + 1]
          enemy.x = currentPath.x + (nextPath.x - currentPath.x) * enemy.progress
          enemy.y = currentPath.y + (nextPath.y - currentPath.y) * enemy.progress
        }
      } else {
        // 敵がゴールに到達
        this.health -= 10
        this.enemies.splice(enemyIndex, 1)
        if (this.health <= 0) {
          this.gameOver = true
        }
      }
    })

    // タワーの攻撃
    this.towers.forEach(tower => {
      if (this.gameTime - tower.lastShot > 0.5) { // 0.5秒間隔で攻撃（高速化）
        const target = this.findNearestEnemy(tower)
        if (target) {
          this.projectiles.push({
            x: tower.x,
            y: tower.y,
            targetX: target.x,
            targetY: target.y,
            damage: tower.damage,
            speed: 5
          })
          tower.lastShot = this.gameTime
        }
      }
    })

    // 弾丸の移動と衝突判定
    this.projectiles.forEach((projectile, projIndex) => {
      const dx = projectile.targetX - projectile.x
      const dy = projectile.targetY - projectile.y
      const distance = Math.hypot(dx, dy)
      
      if (distance < projectile.speed) {
        // 弾丸が目標に到達
        const target = this.enemies.find(enemy => 
          Math.hypot(enemy.x - projectile.targetX, enemy.y - projectile.targetY) < 20
        )
        if (target) {
          target.health -= projectile.damage
          if (target.health <= 0) {
            this.money += GAME_CONFIG.ENEMY_REWARD
            this.score += 100
            const enemyIndex = this.enemies.indexOf(target)
            if (enemyIndex > -1) this.enemies.splice(enemyIndex, 1)
          }
        }
        this.projectiles.splice(projIndex, 1)
      } else {
        projectile.x += (dx / distance) * projectile.speed
        projectile.y += (dy / distance) * projectile.speed
      }
    })
  }

  findNearestEnemy(tower) {
    let nearest = null
    let minDistance = tower.range
    
    this.enemies.forEach(enemy => {
      const distance = Math.hypot(enemy.x - tower.x, enemy.y - tower.y)
      if (distance < minDistance) {
        minDistance = distance
        nearest = enemy
      }
    })
    
    return nearest
  }

  getState() {
    return {
      money: this.money,
      health: this.health,
      wave: this.wave,
      score: this.score,
      towers: this.towers.length,
      enemies: this.enemies.length,
      gameTime: this.gameTime,
      efficiency: this.towers.length > 0 ? this.score / (this.towers.length * GAME_CONFIG.TOWER_COST) : 0,
      survival: this.health / GAME_CONFIG.INITIAL_HEALTH,
      progress: this.wave
    }
  }
}

function App() {
  const canvasRef = useRef(null)
  const gameStateRef = useRef(new GameState())
  const animationRef = useRef(null)
  
  const [gameState, setGameState] = useState(gameStateRef.current.getState())
  const [isLLMEnabled, setIsLLMEnabled] = useState(false)
  const [llmGuidance, setLLMGuidance] = useState({
    recommendation: '',
    reasoning: '',
    priority: 'medium'
  })
  const [gameMode, setGameMode] = useState('manual') // 'manual', 'elm', 'elm_llm'
  const [isGameRunning, setIsGameRunning] = useState(false)

  // ゲームループ
  const gameLoop = useCallback(() => {
    const game = gameStateRef.current
    game.update()
    setGameState(game.getState())
    
    if (game.isGameRunning && !game.gameOver) {
      animationRef.current = requestAnimationFrame(gameLoop)
    }
  }, [])

  // キャンバス描画
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    const game = gameStateRef.current
    
    // 背景をクリア
    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, GAME_CONFIG.CANVAS_WIDTH, GAME_CONFIG.CANVAS_HEIGHT)
    
    // パスを描画
    ctx.strokeStyle = '#444'
    ctx.lineWidth = 30
    ctx.beginPath()
    ctx.moveTo(PATH[0].x, PATH[0].y)
    for (let i = 1; i < PATH.length; i++) {
      ctx.lineTo(PATH[i].x, PATH[i].y)
    }
    ctx.stroke()
    
    // タワーを描画
    game.towers.forEach(tower => {
      // タワーの射程範囲
      ctx.strokeStyle = '#333'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.arc(tower.x, tower.y, tower.range, 0, Math.PI * 2)
      ctx.stroke()
      
      // タワー本体
      ctx.fillStyle = '#4ade80'
      ctx.beginPath()
      ctx.arc(tower.x, tower.y, 15, 0, Math.PI * 2)
      ctx.fill()
    })
    
    // 敵を描画
    game.enemies.forEach(enemy => {
      // 敵本体
      ctx.fillStyle = '#ef4444'
      ctx.beginPath()
      ctx.arc(enemy.x, enemy.y, 12, 0, Math.PI * 2)
      ctx.fill()
      
      // ヘルスバー
      const healthPercent = enemy.health / enemy.maxHealth
      ctx.fillStyle = '#333'
      ctx.fillRect(enemy.x - 15, enemy.y - 20, 30, 4)
      ctx.fillStyle = healthPercent > 0.5 ? '#4ade80' : healthPercent > 0.25 ? '#fbbf24' : '#ef4444'
      ctx.fillRect(enemy.x - 15, enemy.y - 20, 30 * healthPercent, 4)
    })
    
    // 弾丸を描画
    game.projectiles.forEach(projectile => {
      ctx.fillStyle = '#fbbf24'
      ctx.beginPath()
      ctx.arc(projectile.x, projectile.y, 3, 0, Math.PI * 2)
      ctx.fill()
    })
  }, [])

  // ゲーム開始
  const startGame = () => {
    const game = gameStateRef.current
    game.reset()
    game.isGameRunning = true
    game.spawnWave()
    setIsGameRunning(true)
    setGameState(game.getState())
    gameLoop()
  }

  // ゲーム停止
  const stopGame = () => {
    const game = gameStateRef.current
    game.isGameRunning = false
    setIsGameRunning(false)
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }
  }

  // キャンバスクリック処理
  const handleCanvasClick = (event) => {
    if (gameMode !== 'manual' || !isGameRunning) return
    
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    
    const game = gameStateRef.current
    if (game.addTower(x, y)) {
      setGameState(game.getState())
    }
  }

  // 描画の更新
  useEffect(() => {
    draw()
  }, [gameState, draw])

  // LLM指導システム（モック）
  useEffect(() => {
    if (!isLLMEnabled || !isGameRunning) return
    
    const interval = setInterval(() => {
      const state = gameStateRef.current.getState()
      
      // 簡単なルールベースの指導（実際のLLM統合は後で実装）
      let recommendation = ''
      let reasoning = ''
      let priority = 'medium'
      
      if (state.money >= GAME_CONFIG.TOWER_COST * 2 && state.towers < 3) {
        recommendation = 'タワーを追加配置しましょう'
        reasoning = '十分な資金があり、防御力を強化する必要があります'
        priority = 'high'
      } else if (state.enemies > 5 && state.towers < state.wave) {
        recommendation = '緊急でタワーを配置してください'
        reasoning = '敵の数が多く、現在の防御では不十分です'
        priority = 'urgent'
      } else if (state.health < 50) {
        recommendation = '防御を最優先にしてください'
        reasoning = 'ヘルスが危険な状態です'
        priority = 'urgent'
      } else {
        recommendation = '現在の戦略を継続してください'
        reasoning = '良好な状態を維持しています'
        priority = 'low'
      }
      
      setLLMGuidance({ recommendation, reasoning, priority })
    }, 3000)
    
    return () => clearInterval(interval)
  }, [isLLMEnabled, isGameRunning])

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      <div className="max-w-7xl mx-auto">
        {/* ヘッダー */}
        <div className="mb-6">
          <h1 className="text-4xl font-bold text-center mb-2 bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
            Tower Defense LLM Trainer
          </h1>
          <p className="text-center text-gray-400">
            LLM指導型学習システムでタワーディフェンスの戦略を最適化
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* ゲーム画面 */}
          <div className="lg:col-span-3">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  ゲーム画面
                </CardTitle>
              </CardHeader>
              <CardContent>
                <canvas
                  ref={canvasRef}
                  width={GAME_CONFIG.CANVAS_WIDTH}
                  height={GAME_CONFIG.CANVAS_HEIGHT}
                  className="border border-gray-600 rounded-lg cursor-crosshair bg-gray-900"
                  onClick={handleCanvasClick}
                />
                
                {/* ゲーム制御 */}
                <div className="mt-4 flex gap-2">
                  <Button 
                    onClick={startGame} 
                    disabled={isGameRunning}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    ゲーム開始
                  </Button>
                  <Button 
                    onClick={stopGame} 
                    disabled={!isGameRunning}
                    variant="destructive"
                  >
                    ゲーム停止
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* サイドパネル */}
          <div className="space-y-6">
            {/* ゲーム状況 */}
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5" />
                  ゲーム状況
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Coins className="w-4 h-4 text-yellow-500" />
                    資金
                  </span>
                  <span className="font-bold">${gameState.money}</span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Heart className="w-4 h-4 text-red-500" />
                    ヘルス
                  </span>
                  <span className="font-bold">{gameState.health}</span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-blue-500" />
                    ウェーブ
                  </span>
                  <span className="font-bold">{gameState.wave}</span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span>スコア</span>
                  <span className="font-bold">{gameState.score}</span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span>タワー数</span>
                  <span className="font-bold">{gameState.towers}</span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span>敵数</span>
                  <span className="font-bold">{gameState.enemies}</span>
                </div>
              </CardContent>
            </Card>

            {/* LLM指導システム */}
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bot className="w-5 h-5" />
                  LLM戦略指導
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-2">
                  <Button
                    onClick={() => setIsLLMEnabled(!isLLMEnabled)}
                    variant={isLLMEnabled ? "default" : "outline"}
                    size="sm"
                  >
                    {isLLMEnabled ? "ON" : "OFF"}
                  </Button>
                  <span className="text-sm">LLM指導システム</span>
                </div>
                
                {isLLMEnabled && (
                  <div className="space-y-3">
                    <div>
                      <Badge 
                        variant={
                          llmGuidance.priority === 'urgent' ? 'destructive' :
                          llmGuidance.priority === 'high' ? 'default' :
                          'secondary'
                        }
                        className="mb-2"
                      >
                        {llmGuidance.priority === 'urgent' ? '緊急' :
                         llmGuidance.priority === 'high' ? '重要' :
                         llmGuidance.priority === 'medium' ? '中程度' : '低'}
                      </Badge>
                      <p className="text-sm font-medium">{llmGuidance.recommendation}</p>
                    </div>
                    
                    <div>
                      <p className="text-xs text-gray-400 mb-1">理由:</p>
                      <p className="text-xs">{llmGuidance.reasoning}</p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 実験制御 */}
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <User className="w-5 h-5" />
                  実験制御
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button 
                  onClick={() => setGameMode('manual')}
                  variant={gameMode === 'manual' ? 'default' : 'outline'}
                  className="w-full"
                  size="sm"
                >
                  🎮 手動プレイ
                </Button>
                <Button 
                  onClick={() => setGameMode('elm')}
                  variant={gameMode === 'elm' ? 'default' : 'outline'}
                  className="w-full"
                  size="sm"
                >
                  🤖 ELMのみ
                </Button>
                <Button 
                  onClick={() => setGameMode('elm_llm')}
                  variant={gameMode === 'elm_llm' ? 'default' : 'outline'}
                  className="w-full"
                  size="sm"
                >
                  🧠 ELM+LLM指導
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
