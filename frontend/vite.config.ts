import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiTarget = env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000'
  const devHost = env.VITE_DEV_HOST || '0.0.0.0'
  const devPort = Number(env.VITE_DEV_PORT || 3418)
  const publicHost = env.VITE_PUBLIC_HOST || 'cellabsorb.cn'
  const publicProtocol = env.VITE_PUBLIC_PROTOCOL || 'https'
  const publicClientPort = Number(env.VITE_PUBLIC_CLIENT_PORT || (publicProtocol === 'https' ? 443 : 80))
  const enablePublicHmr = env.VITE_PUBLIC_HMR === 'true'
  const allowedHosts = [publicHost, `.${publicHost}`]

  return {
    plugins: [react()],
    build: {
      sourcemap: false,
      chunkSizeWarningLimit: 1000,
      rollupOptions: {
        output: {
          manualChunks: {
            charts: ['recharts'],
            markdown: ['react-markdown', 'remark-gfm', 'marked'],
            qrcode: ['qrcode'],
            pinyin: ['tiny-pinyin'],
            'image-export': ['html2canvas', 'html-to-image'],
            'grid-layout': ['react-grid-layout'],
          },
        },
      },
    },
    server: {
      host: devHost,
      port: devPort,
      strictPort: true,
      allowedHosts,
      hmr: enablePublicHmr
        ? {
            host: publicHost,
            protocol: publicProtocol === 'https' ? 'wss' : 'ws',
            clientPort: publicClientPort,
          }
        : undefined,
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
        },
        '/health': {
          target: apiTarget,
          changeOrigin: true,
        },
      },
    },
    preview: {
      host: devHost,
      port: devPort,
      strictPort: true,
      allowedHosts,
    },
  }
})
