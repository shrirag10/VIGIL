/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        vigil: {
          // CSS variable-driven colors — switch per theme
          bg:           'var(--v-bg)',
          surface:      'var(--v-surface)',
          surface2:     'var(--v-surface2)',
          surface3:     'var(--v-surface3)',
          border:       'var(--v-border)',
          'border-light': 'var(--v-border-light)',
          text:         'var(--v-text)',
          'text-secondary': 'var(--v-text-secondary)',
          dim:          'var(--v-dim)',
          muted:        'var(--v-muted)',
          accent:       'var(--v-accent)',
          'accent-dim': 'var(--v-accent-dim)',
          green:        'var(--v-green)',
          red:          'var(--v-red)',
          blue:         'var(--v-blue)',
          purple:       'var(--v-purple)',
          cyan:         'var(--v-cyan)',
        },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'Fira Code', 'monospace'],
      },
      borderRadius: {
        vigil: '16px',
        'vigil-sm': '10px',
        'vigil-xs': '6px',
      },
      boxShadow: {
        'vigil': '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.06)',
        'vigil-lg': '0 10px 40px rgba(0,0,0,0.15), 0 2px 10px rgba(0,0,0,0.1)',
        'vigil-glow': '0 0 20px var(--v-accent-dim)',
        'vigil-inset': 'inset 0 1px 2px rgba(0,0,0,0.1)',
      },
      animation: {
        'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'blink': 'blink 1s steps(1) infinite',
        'slide-in': 'slideIn 0.3s ease-out',
        'fade-in': 'fadeIn 0.4s ease-out',
        'scale-in': 'scaleIn 0.2s ease-out',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite',
        'count-up': 'countUp 0.4s ease-out',
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' },
        },
        slideIn: {
          from: { transform: 'translateX(40px)', opacity: '0' },
          to: { transform: 'translateX(0)', opacity: '1' },
        },
        fadeIn: {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        scaleIn: {
          from: { opacity: '0', transform: 'scale(0.95)' },
          to: { opacity: '1', transform: 'scale(1)' },
        },
        glowPulse: {
          '0%, 100%': { boxShadow: '0 0 5px var(--v-accent-dim)' },
          '50%': { boxShadow: '0 0 20px var(--v-accent-dim)' },
        },
        countUp: {
          from: { opacity: '0', transform: 'translateY(10px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}
